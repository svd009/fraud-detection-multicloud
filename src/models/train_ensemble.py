import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

print("🚀 XGBoost + RF Fraud Detection Pipeline")
print("📊 Loading data...")

X = pd.read_csv('data/processed/X_train.csv')
y = pd.read_csv('data/processed/y_train.csv')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🤖 Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)

print("🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate
xgb_score = xgb.score(X_val, y_val)
rf_score = rf.score(X_val, y_val)

print(f"\n📈 Production Results:")
print(f"   XGBoost:     {xgb_score:.3f}")
print(f"   RandomForest:{rf_score:.3f}")
print(f"   Best Model:  {'XGBoost' if xgb_score > rf_score else 'RandomForest'}")

joblib.dump(xgb, 'models/xgb_fraud_detector.pkl')
joblib.dump(rf, 'models/rf_fraud_detector.pkl')
print("💾 Models saved - Production ready!")