import pickle
import urllib.request
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
import sklearn

URL = "https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/pipeline_v1.bin"
MODEL_FILE = "pipeline_v1.pkl"

def download_pipeline(url: str = URL, filename: str = MODEL_FILE):
    import os
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    return filename

def load_pipeline(filename: str = MODEL_FILE):
    with open(filename, "rb") as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

dv, model = load_pipeline(download_pipeline())

app = FastAPI(title="Homework 5 API")

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(client: Client):
    """Predict probability for a single client"""
    X = dv.transform([client.dict()])
    y_pred = model.predict_proba(X)[:, 1]
    return {"prediction": float(y_pred[0])}
