# 🌐 Multi-Cloud Fraud Detection Pipeline

A production-grade fraud detection system using **XGBoost + LightGBM** with a target of **97% F1-score**, designed for deployment on **AWS SageMaker** and **GCP Vertex AI**.

## 🎯 Project Goals

- Build a robust fraud detection model on an imbalanced dataset.
- Track experiments and models using **MLflow**.
- Package inference as a **FastAPI** service.
- Prepare deployment scripts for **AWS SageMaker** and **GCP Vertex AI**.

## 🧠 Tech Stack

- Python, Pandas, NumPy, scikit-learn, XGBoost, LightGBM  
- MLflow  
- FastAPI, Uvicorn  
- AWS SDK (boto3), Google Cloud AI Platform SDK  
- Docker, Kubernetes (future)

## 📂 Repository Structure (planned)

- `src/data/preprocess.py` – data loading, cleaning, feature engineering  
- `src/models/train_ensemble.py` – XGBoost + LightGBM training, metrics, MLflow logging  
- `src/deploy/api.py` – FastAPI app for online inference  
- `notebooks/` – EDA and model experimentation  
- `docker/` – Dockerfile and scripts for containerization  
- `tests/` – basic unit tests for data and model code  

## 🚀 Next Steps

- [ ] Add Kaggle credit card fraud dataset loader.
- [ ] Implement preprocessing and SMOTE oversampling.
- [ ] Train baseline models and log to MLflow.
- [ ] Build and expose FastAPI inference endpoint.
- [ ] Write deployment scripts for AWS and GCP.

## 🧾 How to Run (later)

```bash
pip install -r requirements.txt
python src/data/preprocess.py
python src/models/train_ensemble.py
uvicorn src/deploy.api:app --reload
