FROM python:3.11-slim
WORKDIR /app
COPY requirements_api.txt requirements.txt ./
# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*
# Installation Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_api.txt
COPY . .
# Créer dossiers
RUN mkdir -p model logs

# Variables environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/healthz || exit 1

# Démarrage
CMD ["uvicorn", "api_games_plus:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
#CMD ["uvicorn", "api_games_plus:app", "--host", "0.0.0.0", "--port", "8000"]
