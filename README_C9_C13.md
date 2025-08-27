# Quick Start (C9→C13)
1) Copier votre `.env` à la racine (ne pas commit) à partir de `.env.example`.
2) Lancer :
```bash
docker-compose up -d --build
```
- API: http://localhost:8000
- Streamlit: http://localhost:8501
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

3) Tests locaux :
```bash
make test
make cov
```

4) CI: pousser sur `main` pour exécuter lint/tests/build.
