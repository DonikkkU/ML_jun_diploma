import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('event_pred.pkl')
class Form(BaseModel):
    session_id: str
    client_id: str
    visit_number: int
    utm_medium: str
    device_category: str
    device_os: str
    device_brand: str
    device_browser: str
    hour: int
    minute: int
    second: int
    geo_location: str
    device_screen_resolution_area: int


class Prediction(BaseModel):
    session_id: str
    target: float


@app.get("/status")
def status():
    return {"status": "IM OK"}
#
# @app.get("/version")
# def version():
#     return model['metadata']

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    print(form.dict())
    df = pd.DataFrame.from_dict([form.dict()])
    y = model.predict(df)

    return {
        'session_id': form.session_id,
        'target': float(y)
    }
