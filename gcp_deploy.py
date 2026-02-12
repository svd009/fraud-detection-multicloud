from google.cloud import aiplatform
import joblib

aiplatform.init(project='your-gcp-project', location='us-central1')

model = aiplatform.Model.upload(
    display_name="fraud-detection-rf",
    artifact_uri="models/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
)

endpoint = model.deploy(machine_type="n1-standard-4")
