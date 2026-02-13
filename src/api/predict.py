from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI(title="🚀 Fraud Detection API")

# Load XGBoost model
model = joblib.load("models/xgb_fraud_detector.pkl")

@app.get("/")
def root():
    return {"status": "Fraud Detection API Running ✅", "model": "XGBoost Production"}

@app.post("/predict")
def predict_fraud(features: list):
    df = pd.DataFrame([features], columns=[f"feature_{i}" for i in range(20)])
    fraud_prob = model.predict_proba(df)[0][1]
    return {
        "fraud_probability": float(fraud_prob), 
        "is_fraud": fraud_prob > 0.5,
        "confidence": float(fraud_prob)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)