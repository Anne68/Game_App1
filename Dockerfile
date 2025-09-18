FROM python:3.11-slim

WORKDIR /app

# Copier les requirements
COPY requirements_api.txt ./

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_api.txt

# Copier le code
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p model logs compliance

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# CRITIQUE: Utiliser le port dynamique de Render
EXPOSE $PORT

# Health check adapté au port dynamique
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:$PORT/healthz || exit 1

# Commande de démarrage avec port dynamique Render
CMD ["sh", "-c", "uvicorn api_games_plus:app --host 0.0.0.0 --port $PORT --log-level info"]
