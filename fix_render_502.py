#!/usr/bin/env python3
"""
Solution pour erreur 502 Bad Gateway sur Render
Diagnostic et correction des problèmes de démarrage en production
"""

import os
import json
import subprocess
from pathlib import Path

def analyze_502_causes():
    """Analyse les causes possibles de l'erreur 502"""
    print("🔍 ANALYSE ERREUR 502 BAD GATEWAY")
    print("=" * 40)
    
    print("💡 L'erreur 502 indique que:")
    print("   • L'application ne démarre pas correctement sur Render")
    print("   • Le serveur web (uvicorn) ne répond pas sur le bon port")
    print("   • Il y a une erreur lors du startup de l'application")
    print("   • Les dépendances ne s'installent pas correctement")
    print()

def check_render_requirements():
    """Vérifie les exigences spécifiques à Render"""
    print("📋 VÉRIFICATION EXIGENCES RENDER")
    print("=" * 35)
    
    issues = []
    fixes_needed = []
    
    # 1. Vérifier le port dynamique
    print("1️⃣ Configuration du port...")
    dockerfile_path = Path("Dockerfile")
    if dockerfile_path.exists():
        with open(dockerfile_path, "r") as f:
            dockerfile_content = f.read()
        
        if "--port $PORT" not in dockerfile_content and "--port 8000" in dockerfile_content:
            issues.append("Port fixe dans Dockerfile au lieu du port dynamique")
            fixes_needed.append("fix_dockerfile_port")
    
    # 2. Vérifier requirements_api.txt
    print("2️⃣ Vérification des dépendances...")
    req_file = Path("requirements_api.txt")
    if not req_file.exists():
        issues.append("requirements_api.txt manquant")
        fixes_needed.append("create_requirements")
    else:
        with open(req_file, "r") as f:
            requirements = f.read()
        
        essential_deps = ["fastapi", "uvicorn", "pymysql", "scikit-learn"]
        missing_deps = []
        
        for dep in essential_deps:
            if dep not in requirements.lower():
                missing_deps.append(dep)
        
        if missing_deps:
            issues.append(f"Dépendances manquantes: {', '.join(missing_deps)}")
            fixes_needed.append("update_requirements")
    
    # 3. Vérifier structure des modules
    print("3️⃣ Structure des modules...")
    required_files = [
        "api_games_plus.py",
        "settings.py", 
        "model_manager.py",
        "monitoring_metrics.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        issues.append(f"Fichiers manquants: {', '.join(missing_files)}")
    
    # 4. Vérifier compliance
    print("4️⃣ Module compliance...")
    compliance_dir = Path("compliance")
    if not compliance_dir.exists():
        issues.append("Dossier compliance manquant")
        fixes_needed.append("create_compliance")
    else:
        if not (compliance_dir / "__init__.py").exists():
            issues.append("compliance/__init__.py manquant")
            fixes_needed.append("create_compliance")
        
        if not (compliance_dir / "standards_compliance.py").exists():
            issues.append("compliance/standards_compliance.py manquant")
            fixes_needed.append("create_compliance")
    
    return issues, fixes_needed

def fix_dockerfile_port():
    """Corrige le Dockerfile pour utiliser le port dynamique de Render"""
    print("🐳 Correction du Dockerfile...")
    
    dockerfile_content = '''FROM python:3.11-slim

WORKDIR /app

# Copier les requirements
COPY requirements_api.txt requirements.txt ./

# Installation des dépendances système
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
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

# IMPORTANT: Utiliser le port dynamique de Render
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
  CMD curl -f http://localhost:$PORT/healthz || exit 1

# Commande de démarrage avec port dynamique
CMD uvicorn api_games_plus:app --host 0.0.0.0 --port $PORT --log-level info
'''
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Dockerfile corrigé pour port dynamique")
    return True

def create_requirements():
    """Crée requirements_api.txt complet"""
    print("📦 Création de requirements_api.txt...")
    
    requirements_content = '''# FastAPI et serveur
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Authentification
passlib==1.7.4
bcrypt==3.2.2
python-jose[cryptography]==3.3.0

# Base de données
pymysql==1.1.0
python-dotenv==1.0.0
pydantic-settings==2.1.0

# Machine Learning
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.3.2
scipy==1.11.4

# Monitoring
prometheus-client==0.19.0
prometheus-fastapi-instrumentator==6.1.0

# Sécurité et nettoyage
bleach==6.1.0

# Système
psutil==5.9.0

# Build
setuptools>=70
wheel
'''
    
    with open("requirements_api.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ requirements_api.txt créé")
    return True

def create_compliance():
    """Crée le module compliance complet"""
    print("🔒 Création du module compliance...")
    
    # Créer le dossier
    compliance_dir = Path("compliance")
    compliance_dir.mkdir(exist_ok=True)
    
    # __init__.py
    init_content = '''# compliance/__init__.py
"""Module de conformité E4"""
__version__ = "1.0.0"

try:
    from .standards_compliance import SecurityValidator, AccessibilityValidator
    __all__ = ['SecurityValidator', 'AccessibilityValidator']
    COMPLIANCE_AVAILABLE = True
except ImportError:
    __all__ = []
    COMPLIANCE_AVAILABLE = False
'''
    
    # standards_compliance.py
    standards_content = '''# compliance/standards_compliance.py
import re
import html
import logging

logger = logging.getLogger("compliance")

class SecurityValidator:
    def __init__(self):
        self.password_policy = {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digits": True,
        }
    
    def validate_password(self, password: str) -> dict:
        issues = []
        if len(password) < 8:
            issues.append("Password must be at least 8 characters")
        if not re.search(r'[A-Z]', password):
            issues.append("Password must contain uppercase letter")
        if not re.search(r'[a-z]', password):
            issues.append("Password must contain lowercase letter")
        if not re.search(r'\\d', password):
            issues.append("Password must contain digit")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "strength": "Good" if len(issues) == 0 else "Weak"
        }
    
    def sanitize_input(self, input_text: str) -> str:
        if not input_text:
            return ""
        return html.escape(input_text.strip()[:1000])

class AccessibilityValidator:
    def enhance_response_accessibility(self, response: dict) -> dict:
        if not isinstance(response, dict):
            return response
        
        if "recommendations" in response:
            for rec in response["recommendations"]:
                if isinstance(rec, dict) and "title" in rec:
                    rec["aria_label"] = f"Game: {rec['title']}"
        
        response["accessibility"] = {
            "version": "WCAG 2.1 AA",
            "optimized": True
        }
        
        return response
'''
    
    # Écrire les fichiers
    with open(compliance_dir / "__init__.py", "w") as f:
        f.write(init_content)
    
    with open(compliance_dir / "standards_compliance.py", "w") as f:
        f.write(standards_content)
    
    print("✅ Module compliance créé")
    return True

def create_render_start_command():
    """Crée un script de démarrage optimisé pour Render"""
    print("🚀 Création du script de démarrage...")
    
    start_script = '''#!/bin/bash
# start.sh - Script de démarrage pour Render

echo "🚀 Starting Games API on Render..."

# Vérifier les variables d'environnement critiques
if [ -z "$PORT" ]; then
    echo "❌ PORT variable not set"
    exit 1
fi

if [ -z "$SECRET_KEY" ]; then
    echo "⚠️ SECRET_KEY not set, using default (not secure)"
    export SECRET_KEY="render-default-key-not-secure-change-in-production"
fi

# Définir les variables par défaut
export DB_REQUIRED="${DB_REQUIRED:-false}"
export DEMO_LOGIN_ENABLED="${DEMO_LOGIN_ENABLED:-true}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export PYTHONPATH="/app"

echo "📋 Configuration:"
echo "   PORT: $PORT"
echo "   DB_REQUIRED: $DB_REQUIRED"
echo "   DEMO_LOGIN_ENABLED: $DEMO_LOGIN_ENABLED"
echo "   LOG_LEVEL: $LOG_LEVEL"

# Créer les dossiers si nécessaire
mkdir -p model logs

# Test rapide d'import
echo "🧪 Testing imports..."
python -c "
import sys
sys.path.insert(0, '/app')
try:
    import api_games_plus
    print('✅ Import successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
" || exit 1

echo "🎯 Starting uvicorn..."
exec uvicorn api_games_plus:app \\
    --host 0.0.0.0 \\
    --port $PORT \\
    --log-level info \\
    --access-log
'''
    
    with open("start.sh", "w") as f:
        f.write(start_script)
    
    # Rendre le script exécutable
    os.chmod("start.sh", 0o755)
    
    print("✅ Script de démarrage créé")
    return True

def create_updated_dockerfile():
    """Crée un Dockerfile optimisé pour Render"""
    print("🐳 Création du Dockerfile optimisé...")
    
    dockerfile_content = '''FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_api.txt ./

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \\
    && pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p model logs compliance

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Make start script executable
RUN chmod +x start.sh

# Expose port (Render will set $PORT)
EXPOSE $PORT

# Use start script for better error handling
CMD ["./start.sh"]
'''
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Dockerfile optimisé créé")
    return True

def main():
    """Fonction principale de correction 502"""
    print("🚨 CORRECTION ERREUR 502 BAD GATEWAY")
    print("=" * 45)
    
    # Analyser les causes
    analyze_502_causes()
    
    # Vérifier les problèmes
    issues, fixes_needed = check_render_requirements()
    
    if issues:
        print("❌ PROBLÈMES DÉTECTÉS:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print()
    
    # Appliquer les corrections
    print("🔧 APPLICATION DES CORRECTIONS...")
    
    corrections = [
        ("requirements_api.txt", create_requirements),
        ("Module compliance", create_compliance), 
        ("Script de démarrage", create_render_start_command),
        ("Dockerfile optimisé", create_updated_dockerfile)
    ]
    
    success_count = 0
    for name, fix_func in corrections:
        try:
            if fix_func():
                success_count += 1
                print(f"✅ {name}")
            else:
                print(f"❌ {name}")
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    print(f"\n🎯 {success_count}/{len(corrections)} corrections appliquées")
    
    # Instructions de déploiement
    print("\n📋 INSTRUCTIONS DE DÉPLOIEMENT:")
    print("1. Committez tous les changements:")
    print("   git add .")
    print("   git commit -m 'fix: Resolve 502 Bad Gateway - optimize for Render'")
    print("   git push")
    print()
    print("2. Dans Render Dashboard:")
    print("   • Vérifiez que ces variables d'environnement sont définies:")
    print("     - SECRET_KEY (générez une clé sécurisée)")
    print("     - DB_REQUIRED=false")
    print("     - DEMO_LOGIN_ENABLED=true")
    print("     - LOG_LEVEL=INFO")
    print()
    print("3. Forcez un redéploiement:")
    print("   • Cliquez sur 'Manual Deploy' dans Render")
    print("   • Ou poussez un commit vide: git commit --allow-empty -m 'trigger deploy'")
    print()
    print("4. Surveillez les logs de démarrage dans Render")
    print()
    print("🎯 L'erreur 502 devrait être résolue après ces corrections!")

if __name__ == "__main__":
    main()
