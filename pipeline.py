import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score


df_sessions = pd.read_pickle('ga_sessions.pkl')
df_hits = pd.read_pickle('ga_hits.pkl')


class DatasetPreparation:
    def __init__(self):
        self.df_final = None

    def prepare_final_dataset(self):
        columns_to_drop = ['utm_source', 'utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_model']
        df_hits['target'] = df_hits['event_action'].isin(['sub_car_claim_click', 'sub_car_claim_submit_click',
                                                          'sub_open_dialog_click', 'sub_custom_question_submit_click',
                                                          'sub_call_number_click', 'sub_callback_submit_click',
                                                          'sub_submit_success', 'sub_car_request_submit_click']).astype(int)

        n_event = len(df_hits[df_hits['target'] == 1])
        self.df_final = df_sessions.merge(df_hits[['session_id', 'target']], on='session_id', how='left')
        self.df_final = pd.concat([self.df_final[self.df_final['target'] == 1],
                              self.df_final[self.df_final['target'] == 0].sample(n=2 * n_event, random_state=12)]).reset_index(
            drop=True)

        for col in columns_to_drop:
            self.df_final = self.df_final.drop(col, axis=1)

        return self.df_final

    def tranform_data(self):
        self.df_final['visit_date'] = pd.to_datetime(self.df_final['visit_date'])
        self.df_final['visit_time'] = self.df_final['visit_time'].astype(str)

        # Split visit_time and create new features
        self.df_final[['hour', 'minute', 'second']] = self.df_final['visit_time'].str.split(':', expand=True)
        self.df_final[['hour', 'minute', 'second']] = self.df_final[['hour', 'minute', 'second']].astype(int)
        self.df_final = self.df_final.drop('visit_time', axis=1)

        return self.df_final

    def clean_data_devices(self):
        self.df_final.loc[(self.df_final['device_os'].isna()) & (self.df_final['device_screen_resolution'] >= '1920x1080'), 'device_os'] = 'Windows'
        self.df_final.loc[(self.df_final['device_brand'].isna()) & (
                    self.df_final['device_screen_resolution'] >= '1920x1080'), 'device_brand'] = 'Unknown Computer Brand'

        return self.df_final

    def create_new_features(self):
        self.df_final['geo_location'] = self.df_final['geo_country'] + '_' + self.df_final['geo_city']
        self.df_final['device_screen_resolution_area'] = self.df_final['device_screen_resolution'].str.split('x').apply(
            lambda x: int(x[0]) * int(x[1]))
        self.df_final = self.df_final.drop(columns=['geo_country', 'geo_city', 'device_screen_resolution'], axis=1)
        self.df_final.to_pickle('final_data')
        return self.df_final


# Пример использования класса
# dataset_prep = DatasetPreparation()
#
# # Вызов метода prepare_final_dataset
# df_final = dataset_prep.prepare_final_dataset()
# print(df_final['target'].value_counts())
# # Вызов остальных методов класса
# df_final = dataset_prep.tranform_data()
# df_final = dataset_prep.clean_data_devices()
# df_final = dataset_prep.create_new_features()
#
# # Проверка результатов
# print(df_final.head())




class Pipeline_setup:
    def __init__(self):
        self.dataset_prep = DatasetPreparation()
        self.df_final = None

    def process_data(self):
        self.df_final = self.dataset_prep.prepare_final_dataset()
        self.df_final = self.dataset_prep.tranform_data()
        self.df_final = self.dataset_prep.clean_data_devices()
        self.df_final = self.dataset_prep.create_new_features()
        return self.df_final

    def pipeline(self):
        print(self.df_final.dtypes)
        X = self.df_final.drop('target', axis=1)
        y = self.df_final['target']

        numerical_columns = X.select_dtypes(include=['int64', 'float64', 'int32']).columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        categorical_columns = categorical_columns[~categorical_columns.isin(['session_id', 'client_id'])]

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('numerical', numerical_transformer, numerical_columns),
            ('categorical', categorical_transformer, categorical_columns)
        ])

        models = (
        LogisticRegression(penalty='l2', C=0.01, solver='liblinear', random_state=123),
        RandomForestClassifier(n_estimators=100, max_depth=None,  random_state=42, min_samples_leaf=3, min_samples_split=2)
        )

        best_score = .0
        best_pipe = None
        for model in models:
            pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            pipe.fit(X, y)
            predictions = cross_val_predict(pipe, X, y, cv=4, method='predict_proba')
            print(f'model: {type(model).__name__}, auc: {predictions}')
            score = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
            print(f'model: {type(model).__name__}, auc_mean: {score.mean():.4f}, auc_std: {score.std():.4f}')
            if score.mean() > best_score:
                best_score = score.mean()
                best_pipe = pipe

        print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
        joblib.dump(best_pipe, 'event_pred.pkl')

#Pipeline
if __name__ == '__main__':
    pipeline_setup = Pipeline_setup()
    pipeline_setup.process_data()
    pipeline_setup.pipeline()