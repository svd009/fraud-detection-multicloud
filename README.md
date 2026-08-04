# Fraud Detection Pipeline (Multi-Cloud)

End-to-end machine learning pipeline for credit card fraud detection. Trains XGBoost and Random Forest classifiers on transaction data, exposes predictions through a FastAPI REST API with interactive Swagger docs, and ships with deployment scripts for AWS and GCP.

## Overview

Fraud detection is a highly imbalanced classification problem — fraudulent transactions typically make up well under 1% of all transactions. This project builds a practical pipeline for that problem:

- Preprocesses and engineers features from raw transaction data
- Trains an ensemble of models (XGBoost, Random Forest) tuned for class imbalance
- Serves real-time predictions through a FastAPI endpoint with auto-generated Swagger/OpenAPI docs
- Supports containerized deployment (Docker) and includes helper scripts for AWS and GCP

## Features

- Automated data preprocessing and feature engineering (`src/data_preprocessor.py`)
- Ensemble modeling with XGBoost and Random Forest, with `scale_pos_weight` tuning to handle severe class imbalance
- Modular training pipeline (`src/model_trainer.py`, `src/models/train_ensemble.py`)
- Real-time inference API built with FastAPI (`src/deploy/api.py`, `src/api/predict.py`)
- Interactive Swagger UI for testing predictions without writing client code
- Docker support for containerized deployment
- Deployment helper scripts for AWS (`aws_deploy.py`) and GCP (`gcp_deploy.py`)
- Model persistence with `joblib` (trained models and scaler saved under `models/`)

## Tech Stack

| Category | Tools |
|---|---|
| Modeling | XGBoost, scikit-learn (Random Forest), LightGBM, imbalanced-learn |
| API | FastAPI, Uvicorn |
| Data | pandas, NumPy |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| Deployment | Docker, Kubernetes, AWS (boto3), GCP (Vertex AI) |
| Dashboard (optional) | Streamlit |

## Project Structure

```
fraud-detection-multicloud/
├── data/
│   ├── raw/                  # Original transaction dataset
│   └── processed/            # Train-ready features and labels
├── models/                   # Saved model artifacts (.pkl)
├── notebooks/                # Exploratory analysis
├── src/
│   ├── data_preprocessor.py  # Feature engineering / preprocessing
│   ├── model_trainer.py      # Core training logic
│   ├── main.py                # Pipeline entry point
│   ├── api/
│   │   └── predict.py        # Prediction request handling
│   ├── deploy/
│   │   └── api.py            # FastAPI app definition
│   └── models/
│       └── train_ensemble.py # Ensemble model training
├── tests/                    # Test suite
├── train_baseline.py         # Quick XGBoost baseline trainer
├── run_api.py                # Local API launcher (uvicorn)
├── download_data.py          # Dataset download helper
├── aws_deploy.py              # AWS deployment helper
├── gcp_deploy.py               # GCP deployment helper
├── Dockerfile
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/svd009/fraud-detection-multicloud.git
cd fraud-detection-multicloud
pip install -r requirements.txt
```

### Get the data

```bash
python download_data.py
```

<!-- CONFIRM: what does download_data.py actually pull, and from where (e.g. Kaggle credit card fraud dataset)? -->

### Train a model

Quick baseline (XGBoost only):

```bash
python train_baseline.py
```

Full ensemble (XGBoost + Random Forest):

```bash
python -m src.models.train_ensemble
```

<!-- CONFIRM: exact command/entry point for the ensemble trainer, and any config flags it accepts -->

Trained models are saved to `models/` as `.pkl` files and loaded by the API at inference time.

### Run the API locally

```bash
python run_api.py
```

The API will be available at `http://127.0.0.1:8000`. Interactive Swagger docs are at `http://127.0.0.1:8000/docs`.

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"...": "..."}'
```

<!-- CONFIRM: actual endpoint path and request/response schema from src/deploy/api.py and src/api/predict.py -->

### Run with Docker

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

<!-- FLAG: Dockerfile CMD currently points to "main:app", but run_api.py and the module layout point to "src.deploy.api:app". Worth double-checking the Dockerfile CMD before relying on this section. -->

## Model Performance

<!-- CONFIRM: paste your actual evaluation numbers here (Accuracy, Precision, Recall, F1, ROC-AUC on the held-out test set). train_baseline.py prints an accuracy score but the "97%+ F1" figure in the script's log message looks like a target/comment rather than a verified result — I didn't want to state it as fact without your confirmation. -->

| Metric | Score |
|---|---|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 Score | TBD |
| ROC-AUC | TBD |

## Deployment

The project includes standalone scripts for pushing trained models to cloud ML platforms:

- `aws_deploy.py` — deploy to AWS (SageMaker)
- `gcp_deploy.py` — deploy to GCP (Vertex AI)

<!-- CONFIRM: brief description of what each script actually does (e.g. uploads model to S3 + registers a SageMaker endpoint) -->

## Data

<!-- FLAG: the previous README described this as using "synthetic data," but train_baseline.py reads data/raw/creditcard.csv and handles a ~0.17% fraud rate — this matches the well-known Kaggle "Credit Card Fraud Detection" dataset (real, anonymized transactions), not synthetic data. Let me know which is accurate so I can describe the dataset correctly and credit the source if needed. -->

## Roadmap / Future Work

<!-- CONFIRM: anything you'd like listed here — e.g. Azure ML support, model monitoring, CI/CD, a Streamlit dashboard front-end -->

## License

<!-- CONFIRM: license (MIT is a common default for portfolio projects if you don't already have one) -->

## Author

**Suujay** — [github.com/svd009](https://github.com/svd009)
