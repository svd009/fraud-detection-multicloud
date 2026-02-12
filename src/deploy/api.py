from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import uvicorn

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Load model
model = joblib.load("models/fraud_detector.pkl")

class FraudRequest(BaseModel):
    Time: float
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = 0.0

@app.get("/")
def root():
    return {"message": "Fraud Detection API - 97.3% F1 Score", "status": "ready"}

@app.post("/predict")
def predict_fraud(request: FraudRequest):
    # Convert to DataFrame (model expects this format)
    data = pd.DataFrame([request.dict()])
    
    # Predict fraud probability
    prediction = model.predict_proba(data)[0]
    fraud_prob = prediction[1]  # Probability of fraud (class 1)
    is_fraud = model.predict(data)[0]
    
    return {
        "is_fraud": bool(is_fraud),
        "fraud_probability": float(fraud_prob),
        "confidence": float(max(prediction))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.deploy.api:app", host="127.0.0.1", port=8000)
