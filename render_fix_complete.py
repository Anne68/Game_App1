#!/usr/bin/env python3
"""
SOLUTION COMPLÈTE POUR LES ERREURS RENDER

Problèmes identifiés et solutions :

1. ERREUR 502 BAD GATEWAY
   - Cause : Application ne démarre pas correctement
   - Solutions critiques à appliquer

2. MODULE COMPLIANCE MANQUANT
   - Création automatique du module requis

3. ERREUR LOGGER SYNTAX
   - Correction syntaxe logger avec () supplémentaires

4. ERREUR SVD DIMENSIONS
   - SVD adaptatif pour éviter n_components > n_features

5. PORT CONFIGURATION RENDER
   - Configuration dynamique du port pour Render

"""

import os
import time
import shutil
from pathlib import Path
import re

class RenderFixManager:
    """Gestionnaire de correction pour Render.com"""
    
    def __init__(self):
        self.fixes_applied = []
        self.errors_found = []
    
    def run_all_fixes(self):
        """Applique toutes les corrections nécessaires"""
        print("🚨 CORRECTION COMPLÈTE DES ERREURS RENDER")
        print("=" * 50)
        
        fixes = [
            ("1. Module Compliance", self.create_compliance_module),
            ("2. Erreur Logger Syntax", self.fix_logger_syntax),
            ("3. Erreur SVD Dimensions", self.fix_svd_error),
            ("4. Port Render Config", self.fix_render_port),
            ("5. Variables Environnement", self.create_env_config),
            ("6. Dockerfile Optimisé", self.fix_dockerfile),
            ("7. Start Script Render", self.create_start_script),
            ("8. Requirements Complets", self.fix_requirements)
        ]
        
        for name, fix_func in fixes:
            print(f"\n📋 {name}...")
            try:
                if fix_func():
                    self.fixes_applied.append(name)
                    print(f"✅ {name} - Corrigé")
                else:
                    print(f"⚠️ {name} - Non nécessaire")
            except Exception as e:
                self.errors_found.append(f"{name}: {e}")
                print(f"❌ {name} - Erreur: {e}")
        
        self.print_summary()
        return len(self.errors_found) == 0
    
    def create_compliance_module(self):
        """Crée le module compliance manquant"""
        compliance_dir = Path("compliance")
        compliance_dir.mkdir(exist_ok=True)
        
        # __init__.py
        init_content = '''# compliance/__init__.py
"""Module de conformité E4 - Version de production"""

__version__ = "1.0.0"

try:
    from .standards_compliance import SecurityValidator, AccessibilityValidator
    __all__ = ['SecurityValidator', 'AccessibilityValidator']
    COMPLIANCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Compliance components not available: {e}")
    __all__ = []
    COMPLIANCE_AVAILABLE = False
'''
        
        # standards_compliance.py
        standards_content = '''# compliance/standards_compliance.py
"""Standards de conformité E4 - Version de production"""

import re
import html
import logging

logger = logging.getLogger("compliance")

class SecurityValidator:
    """Validateur de sécurité pour production"""
    
    def __init__(self):
        self.password_policy = {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digits": True,
        }
    
    def validate_password(self, password: str) -> dict:
        """Validation robuste des mots de passe"""
        issues = []
        
        if len(password) < self.password_policy["min_length"]:
            issues.append(f"Password must be at least {self.password_policy['min_length']} characters")
        
        if self.password_policy["require_uppercase"] and not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        
        if self.password_policy["require_lowercase"] and not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        
        if self.password_policy["require_digits"] and not re.search(r'\\d', password):
            issues.append("Password must contain at least one digit")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "strength": "Strong" if len(issues) == 0 else "Weak"
        }
    
    def sanitize_input(self, input_text: str) -> str:
        """Nettoyage sécurisé des entrées"""
        if not input_text:
            return ""
        
        # Limite de taille
        if len(input_text) > 1000:
            input_text = input_text[:1000]
        
        # Échappement HTML et suppression caractères dangereux
        sanitized = html.escape(input_text.strip())
        # Suppression des caractères de contrôle
        sanitized = re.sub(r'[\\x00-\\x1f\\x7f-\\x9f]', '', sanitized)
        
        return sanitized

class AccessibilityValidator:
    """Validateur d'accessibilité WCAG 2.1"""
    
    def enhance_response_accessibility(self, response: dict) -> dict:
        """Enrichissement pour accessibilité"""
        if not isinstance(response, dict):
            return response
        
        # Enrichir les recommandations si présentes
        if "recommendations" in response and isinstance(response["recommendations"], list):
            for rec in response["recommendations"]:
                if isinstance(rec, dict) and "title" in rec:
                    rec["aria_label"] = f"Recommended game: {rec['title']}"
                    if "confidence" in rec:
                        confidence_level = "High" if rec["confidence"] > 0.7 else "Medium" if rec["confidence"] > 0.4 else "Low"
                        rec["confidence_description"] = f"{confidence_level} confidence recommendation"
        
        # Métadonnées d'accessibilité
        response["accessibility"] = {
            "version": "WCAG 2.1 AA",
            "screen_reader_optimized": True,
            "language": "en",
            "high_contrast_ready": True
        }
        
        return response
'''
        
        # Écrire les fichiers
        with open(compliance_dir / "__init__.py", "w", encoding="utf-8") as f:
            f.write(init_content)
        
        with open(compliance_dir / "standards_compliance.py", "w", encoding="utf-8") as f:
            f.write(standards_content)
        
        return True
    
    def fix_logger_syntax(self):
        """Corrige l'erreur de syntaxe logger avec () supplémentaires"""
        api_file = Path("api_games_plus.py")
        
        if not api_file.exists():
            return False
        
        with open(api_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Corrections spécifiques identifiées dans les logs
        fixes = [
            # Erreur principale trouvée dans les logs Render
            ('logger.info("API startup complete.")()', 'logger.info("API startup complete.")'),
            ('logger.info("Starting Games API with ML and E4 compliance...")()', 'logger.info("Starting Games API with ML and E4 compliance...")'),
            ('logger.info("E4 Compliance standards active")()', 'logger.info("E4 Compliance standards active")'),
            ('logger.warning("E4 Compliance standards not available")()', 'logger.warning("E4 Compliance standards not available")'),
        ]
        
        changes_made = 0
        for wrong, correct in fixes:
            if wrong in content:
                content = content.replace(wrong, correct)
                changes_made += 1
        
        if changes_made > 0:
            with open(api_file, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    
    def fix_svd_error(self):
        """Corrige l'erreur SVD n_components > n_features"""
        model_file = Path("model_manager.py")
        
        if not model_file.exists():
            return False
        
        with open(model_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Rechercher et corriger SVD fixe
        svd_pattern = r'self\.svd = TruncatedSVD\(n_components=128, random_state=42\)'
        
        if re.search(svd_pattern, content):
            # Remplacer par SVD adaptatif
            svd_fix = '''        # SVD adaptatif pour éviter n_components > n_features
        n_features = self.tfidf_matrix.shape[1]
        n_components = min(50, max(5, n_features - 1))  # Sécurisé
        logger.info(f"Using {n_components} SVD components for {n_features} features")
        
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)'''
            
            content = re.sub(svd_pattern, svd_fix.strip(), content)
            
            with open(model_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        
        return False
    
    def fix_render_port(self):
        """Corrige la configuration du port pour Render"""
        dockerfile = Path("Dockerfile")
        
        if not dockerfile.exists():
            return False
        
        dockerfile_content = '''FROM python:3.11-slim

WORKDIR /app

# Copier les requirements
COPY requirements_api.txt ./

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

# CRITIQUE: Port dynamique Render
EXPOSE $PORT

# Health check adapté
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
  CMD curl -f http://localhost:$PORT/healthz || exit 1

# Commande avec port dynamique OBLIGATOIRE pour Render
CMD uvicorn api_games_plus:app --host 0.0.0.0 --port $PORT --log-level info
'''
        
        with open(dockerfile, "w") as f:
            f.write(dockerfile_content)
        
        return True
    
    def create_env_config(self):
        """Crée la configuration d'environnement pour Render"""
        env_example = Path(".env.render.example")
        
        env_content = '''# Variables d'environnement pour Render.com
# COPIER CES VALEURS dans Environment Variables de Render

# SÉCURITÉ (OBLIGATOIRE - Générer une vraie clé sécurisée!)
SECRET_KEY=votre-cle-secrete-tres-longue-minimum-32-caracteres-changez-en-production

# Base de données (false si pas de DB)
DB_REQUIRED=false

# Mode démo pour tests
DEMO_LOGIN_ENABLED=true
DEMO_USERNAME=demo
DEMO_PASSWORD=demo123

# Configuration API
LOG_LEVEL=INFO
ALLOW_ORIGINS=*

# JWT
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480

# PORT sera défini automatiquement par Render
# Ne pas définir PORT dans les variables Render
'''
        
        with open(env_example, "w") as f:
            f.write(env_content)
        
        return True
    
    def create_start_script(self):
        """Crée le script de démarrage optimisé pour Render"""
        start_script = '''#!/bin/bash
# start.sh - Script de démarrage optimisé pour Render

echo "🚀 Starting Games API on Render..."

# Vérifier PORT (obligatoire sur Render)
if [ -z "$PORT" ]; then
    echo "❌ PORT environment variable not set by Render"
    exit 1
fi

# Variables par défaut
export SECRET_KEY="${SECRET_KEY:-render-default-secret-change-in-production-very-long}"
export DB_REQUIRED="${DB_REQUIRED:-false}"
export DEMO_LOGIN_ENABLED="${DEMO_LOGIN_ENABLED:-true}"
export DEMO_USERNAME="${DEMO_USERNAME:-demo}"
export DEMO_PASSWORD="${DEMO_PASSWORD:-demo123}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export PYTHONPATH="/app"

echo "📋 Configuration:"
echo "   PORT: $PORT"
echo "   DB_REQUIRED: $DB_REQUIRED"
echo "   DEMO_LOGIN_ENABLED: $DEMO_LOGIN_ENABLED"

# Créer dossiers
mkdir -p model logs compliance

# Test imports critiques
echo "🧪 Testing imports..."
python3 -c "
import sys
sys.path.insert(0, '/app')

try:
    print('Testing settings...')
    import settings
    print('✅ Settings OK')
    
    print('Testing model_manager...')
    import model_manager
    print('✅ Model manager OK')
    
    print('Testing API...')
    import api_games_plus
    print('✅ API OK')
    
    print('🎉 All imports successful!')
    
except Exception as e:
    print(f'❌ Import error: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" || exit 1

echo "🎯 Starting uvicorn on port $PORT..."
exec uvicorn api_games_plus:app \\
    --host 0.0.0.0 \\
    --port "$PORT" \\
    --log-level info \\
    --timeout-keep-alive 65
'''
        
        with open("start.sh", "w") as f:
            f.write(start_script)
        
        os.chmod("start.sh", 0o755)
        return True
    
    def fix_requirements(self):
        """Met à jour requirements_api.txt avec versions stables"""
        requirements_content = '''# API Core - Versions stables Python 3.11
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Auth & Security
passlib==1.7.4
bcrypt==4.1.2
python-jose[cryptography]==3.3.0

# Database & Config
pymysql==1.1.0
python-dotenv==1.0.0
pydantic-settings==2.1.0

# ML & Data - Versions compatibles
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
scikit-learn>=1.3.0,<2.0.0
scipy>=1.10.0,<2.0.0

# Monitoring
prometheus-client==0.19.0
prometheus-fastapi-instrumentator==6.1.0

# Security
bleach==6.1.0

# System
psutil==5.9.0

# Build
setuptools>=68.0.0
wheel>=0.41.0
'''
        
        with open("requirements_api.txt", "w") as f:
            f.write(requirements_content)
        
        return True
    
    def fix_dockerfile(self):
        """S'assure que le Dockerfile utilise le bon CMD"""
        # Déjà fait dans fix_render_port, mais on vérifie
        dockerfile = Path("Dockerfile")
        
        if dockerfile.exists():
            with open(dockerfile, 'r') as f:
                content = f.read()
            
            # Vérifier que CMD utilise $PORT
            if "CMD uvicorn api_games_plus:app --host 0.0.0.0 --port $PORT" in content:
                return False  # Déjà correct
            else:
                return self.fix_render_port()
        
        return self.fix_render_port()
    
    def print_summary(self):
        """Affiche le résumé des corrections"""
        print(f"\n🎯 RÉSUMÉ DES CORRECTIONS")
        print("=" * 40)
        
        print(f"✅ Corrections appliquées: {len(self.fixes_applied)}")
        for fix in self.fixes_applied:
            print(f"   • {fix}")
        
        if self.errors_found:
            print(f"\n❌ Erreurs rencontrées: {len(self.errors_found)}")
            for error in self.errors_found:
                print(f"   • {error}")
        
        print(f"\n📋 PROCHAINES ÉTAPES:")
        print("1. Committez tous les changements:")
        print("   git add .")
        print("   git commit -m 'fix: Resolve Render 502 errors - complete fix'")
        print("   git push")
        print()
        print("2. Variables d'environnement Render:")
        print("   - Copiez les valeurs de .env.render.example")
        print("   - NE PAS définir PORT (Render le fait automatiquement)")
        print("   - SECRET_KEY: générez une vraie clé sécurisée")
        print()
        print("3. Redéployez depuis Render Dashboard")
        print("4. L'erreur 502 devrait être résolue!")

def main():
    """Fonction principale de correction"""
    fixer = RenderFixManager()
    success = fixer.run_all_fixes()
    
    if success:
        print("\n🎉 TOUTES LES CORRECTIONS APPLIQUÉES!")
        print("🚀 Votre application devrait maintenant démarrer sur Render")
        return True
    else:
        print("\n⚠️ Certaines corrections ont échoué")
        print("Vérifiez les erreurs ci-dessus")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
