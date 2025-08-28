# API Game — E3 (C9→C13)

Projet FastAPI prêt pour la **MISE EN SITUATION 2 (E3)**, couvrant les compétences **C9 à C13** :
- C9: API REST sécurisée exposant un modèle d'IA
- C10: Intégration accessible (front/app)
- C11: Monitoring (Prometheus + MLflow)
- C12: Tests automatisés (data → features → modèle → API)
- C13: CI/CD (lint, tests, docker, déploiement)

## Démarrage rapide

```bash
make install
make run  # http://127.0.0.1:8000/docs
```

Ou via Docker:

```bash
docker compose up --build
```

## Endpoints

- `POST /token` : OAuth2 (Password) → JWT
- `POST /games/predict` : prédiction (exemple)
- `GET /healthz` : liveness
- `GET /ready` : readiness
- `GET /metrics` : Prometheus (si activé)

## Sécurité
- Variables d'environnement via `.env` (n'utilise **jamais** des secrets en clair).
- JWT (HS256), CORS restrictif, validation Pydantic stricte.

## Monitoring
- `prometheus_fastapi_instrumentator` expose `/metrics`.
- MLflow journalise les paramètres/métriques d'inférence.

## Tests
```bash
make test
```

## CI/CD
- GitHub Actions: lint + tests + build Docker + (exemple) déclenchement de déploiement.

## Licence
MIT
