import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import logging

class FraudDataPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.time_features = None
        
    def load_kaggle_data(self, filepath):
        """Load Kaggle credit card fraud dataset"""
        logging.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Anonymized features V1-V28 + Time + Amount + Class
        print(f"Dataset shape: {df.shape}")
        print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean():.2%})")
        
        return df
    
    def preprocess(self, df):
        """Handle class imbalance, scaling, feature engineering"""
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Scale Time and Amount (only non-anonymized features)
        time_amount = X[['Time', 'Amount']].copy()
        X_scaled = X.drop(['Time', 'Amount'], axis=1)
        X_scaled[['Time', 'Amount']] = self.scaler.fit_transform(time_amount)
        
        # Train/validation split (80/20)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.time_features = ['Time', 'Amount'] + [f'V{i}' for i in range(1,29)]
        return (X_train, X_val, y_train, y_val)
