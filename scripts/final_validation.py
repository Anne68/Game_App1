# Une fois le pipeline terminé
python scripts/final_validation.py --prod-url https://game-app-y8be.onrender.com

# Vérifier le monitoring
curl https://game-app-y8be.onrender.com/metrics
curl https://game-app-y8be.onrender.com/healthz
