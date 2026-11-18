import os
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Defaults to the exported model shipped with the project, so the API runs without an MLflow server.
# Set MODEL_URI (e.g. "models:/<model-id>") and TRACKING_URI to load a model from MLflow instead.
TRACKING_URI = os.getenv("TRACKING_URI", "http://localhost:5001")
MODEL_URI = os.getenv("MODEL_URI", str(PROJECT_ROOT / "artifacts" / "model"))


def load_model(tracking_uri=TRACKING_URI, model_uri=MODEL_URI):
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(model_uri)
    return model
