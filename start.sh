#!/bin/bash
# start.sh - Script de démarrage optimisé pour Render

echo "🚀 Starting Games API on Render..."

# Vérifier les variables d'environnement critiques
if [ -z "$PORT" ]; then
    echo "❌ PORT variable not set by Render"
    exit 1
fi

if [ -z "$SECRET_KEY" ]; then
    echo "⚠️ SECRET_KEY not set, using default (CHANGE IN PRODUCTION)"
    export SECRET_KEY="render-default-key-not-secure-change-in-production-very-long-secret"
fi

# Définir les variables par défaut pour éviter les erreurs
export DB_REQUIRED="${DB_REQUIRED:-false}"
export DEMO_LOGIN_ENABLED="${DEMO_LOGIN_ENABLED:-true}"
export DEMO_USERNAME="${DEMO_USERNAME:-demo}"
export DEMO_PASSWORD="${DEMO_PASSWORD:-demo123}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export PYTHONPATH="/app"

echo "📋 Configuration Render:"
echo "   PORT: $PORT"
echo "   DB_REQUIRED: $DB_REQUIRED"
echo "   DEMO_LOGIN_ENABLED: $DEMO_LOGIN_ENABLED"
echo "   LOG_LEVEL: $LOG_LEVEL"

# Créer les dossiers si nécessaire
mkdir -p model logs compliance

# Test rapide d'import pour détecter les erreurs tôt
echo "🧪 Testing critical imports..."
python3 -c "
import sys
sys.path.insert(0, '/app')
try:
    print('Testing settings import...')
    import settings
    print('✅ Settings OK')
    
    print('Testing model_manager import...')
    import model_manager
    print('✅ Model manager OK')
    
    print('Testing main app import...')
    import api_games_plus
    print('✅ Main API OK')
    
    print('🎉 All imports successful!')
    
except Exception as e:
    print(f'❌ Import failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" || exit 1

echo "🎯 Starting uvicorn on port $PORT..."
exec uvicorn api_games_plus:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info \
    --access-log \
    --timeout-keep-alive 65
