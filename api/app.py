import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()


_origins_env = os.getenv('FRONTEND_ORIGINS', 'http://localhost:5173')
ORIGINS = [o.strip() for o in _origins_env.split(',') if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: list[float]

CLASSIFIER_PATH = str(os.getenv('CLASSIFIER_PATH'))
clf = pickle.load(open(CLASSIFIER_PATH, 'rb'))

@app.post('/predict')
def predict(req: PredictRequest):
    arr = np.array(req.features).reshape(1, -1)
    pred = int(clf.predict(arr)[0])
    proba = float(clf.predict_proba(arr).max())
    
    return {
        "prediction": pred,
        "probability": proba,
    }

