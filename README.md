# Games API – Pack C9→C13

Ce pack ajoute sécurité (JWT, rate limiting), monitoring Prometheus, tests Pytest,
un client Streamlit, la conteneurisation Docker et un pipeline CI.

## Démarrage rapide (local)
```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn api_games_plus:app --reload
```

Swagger: http://127.0.0.1:8000/docs

Client Streamlit:
```bash
streamlit run client_streamlit.py
```

Tests:
```bash
pytest -q
```

Docker:
```bash
docker build -t games-api:latest .
docker run -p 8000:8000 --env-file .env games-api:latest
```

## Schéma SQL minimal
Voir `sql/schema.sql` pour créer les tables `users`, `games` et `user_game_searches`.
