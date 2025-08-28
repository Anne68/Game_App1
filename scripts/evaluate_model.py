# Exemple: évaluer le modèle et logger les métriques dans MLflow
import mlflow
mlflow.set_experiment("api-game-eval")
with mlflow.start_run(run_name="eval"):
    mlflow.log_metric("dummy_f1", 0.75)
    mlflow.log_param("model", "DummyModel")
print("Evaluation done (dummy).")
