import mlflow
from app.config import settings

def log_inference(metrics: dict, params: dict | None = None, tags: dict | None = None):
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI or "mlruns")
    with mlflow.start_run(run_name="inference", nested=True):
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        if tags:
            mlflow.set_tags(tags)
