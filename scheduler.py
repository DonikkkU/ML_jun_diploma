import joblib
from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
from datetime import datetime
import pandas as pd

sched = BlockingScheduler(timezone=tzlocal.get_localzone())

df = pd.read_pickle('final_data')
pipeline = joblib.load('event_pred.pkl')
print(type(pipeline))

@sched.scheduled_job('cron', second='*/10')
def on_time():
    data = df.sample(frac=0.05)
    target = pipeline.predict(data)  # Predict directly from the loaded pipeline
    data['target'] = target
    print(data[['session_id', 'target']])

if __name__ == '__main__':
    sched.start()



