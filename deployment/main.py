from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Fraud Detection API")

# Load model
MODEL_PATH = "models/fraud_model.pkl"
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        X = np.array(request.features).reshape(1, -1)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]
        return {
            "prediction": int(prediction),
            "fraud_probability": round(float(proba), 4)
        }
    except Exception as e:
        return {"error": str(e)}
