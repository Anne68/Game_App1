# API Game — Render (Python 3.11, no pandas)
- `runtime.txt` force Python 3.11.9
- `mlflow-skinny` remplace `mlflow` pour éviter la dépendance à `pandas`
- Build: `pip install -r requirements.txt`
- Start: `uvicorn api_games_plus:app --host 0.0.0.0 --port 10000`
