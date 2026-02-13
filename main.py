from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

class InputData(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    prediction = model.predict(arr)
    return {"prediction": prediction.tolist()}