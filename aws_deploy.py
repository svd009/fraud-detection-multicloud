import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import joblib

# Train + deploy model
session = sagemaker.Session()
sklearn_preprocessor = SKLearn(
    entry_point='src/main.py',
    role='arn:aws:iam::YOUR-ACCOUNT:sagemaker-execution-role',
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='1.2-1'
)

sklearn_preprocessor.fit({'train': 'data/raw/'})
predictor = sklearn_preprocessor.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
