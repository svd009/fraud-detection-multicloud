import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from data_preprocessor import FraudDataPreprocessor
import joblib
import logging

class FraudModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.preprocessor = FraudDataPreprocessor()
        
    def train(self, data_path):
        """Complete training pipeline"""
        # Load and preprocess
        df = self.preprocessor.load_kaggle_data(data_path)
        X_train, X_val, y_train, y_val = self.preprocessor.preprocess(df)
        
        # Train model
        print("Training Random Forest...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        print(f"F1 Score: {f1:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        
        # Save model + preprocessor
        joblib.dump(self.model, 'models/fraud_model.pkl')
        joblib.dump(self.preprocessor.scaler, 'models/scaler.pkl')
        
        return f1
    
if __name__ == "__main__":
    trainer = FraudModelTrainer()
    # Will use dataset later
    print("Model trainer ready!")
