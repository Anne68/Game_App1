FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api_games_plus.py settings.py ./
EXPOSE 8080
CMD ["uvicorn", "api_games_plus:app", "--host", "0.0.0.0", "--port", "8080"]
