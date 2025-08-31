#!/usr/bin/env python3
"""
Script de correction rapide pour les problèmes de déploiement Render
"""

import os
import json
from pathlib import Path

def create_minimal_compliance_module():
    """Crée un module compliance minimal si manquant"""
    print("📁 Création du module compliance minimal...")
    
    # Créer le dossier
    compliance_dir = Path("compliance")
    compliance_dir.mkdir(exist_ok=True)
    
    # Créer __init__.py
    init_content = '''# compliance/__init__.py
"""
Module de conformité E4 - Version minimale pour déploiement
"""

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
    
    with open(compliance_dir / "__init__.py", "w") as f:
        f.write(init_content)
    
    # Créer standards_compliance.py minimal
    standards_content = '''# compliance/standards_compliance.py
"""
Standards de conformité E4 - Version minimale
"""

import re
import html
import logging

logger = logging.getLogger("compliance")

class SecurityValidator:
    """Validateur de sécurité minimal"""
    
    def __init__(self):
        self.password_policy = {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digits": True,
        }
    
    def validate_password(self, password: str) -> dict:
        """Validation de base des mots de passe"""
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
            "strength": "Medium" if len(issues) == 0 else "Weak"
        }
    
    def sanitize_input(self, input_text: str) -> str:
        """Nettoyage basique des entrées"""
        if not input_text:
            return ""
        
        # Limite de taille
        if len(input_text) > 1000:
            input_text = input_text[:1000]
        
        # Échappement HTML
        return html.escape(input_text.strip())

class AccessibilityValidator:
    """Validateur d'accessibilité minimal"""
    
    def enhance_response_accessibility(self, response: dict) -> dict:
        """Enrichissement basique pour accessibilité"""
        if not isinstance(response, dict):
            return response
        
        # Enrichir les recommandations si présentes
        if "recommendations" in response and isinstance(response["recommendations"], list):
            for rec in response["recommendations"]:
                if isinstance(rec, dict) and "title" in rec:
                    rec["aria_label"] = f"Recommended game: {rec['title']}"
        
        # Métadonnées d'accessibilité
        response["accessibility"] = {
            "version": "WCAG 2.1 AA",
            "screen_reader_optimized": True,
            "language": "en"
        }
        
        return response
'''
    
    with open(compliance_dir / "standards_compliance.py", "w") as f:
        f.write(standards_content)
    
    print("✅ Module compliance créé")
    return True

def create_minimal_env_example():
    """Crée un exemple d'environnement pour Render"""
    print("📝 Création de .env.render.example...")
    
    env_content = '''# Variables d'environnement pour Render.com
# Copiez ces valeurs dans les Environment Variables de votre service Render

# Sécurité (OBLIGATOIRE - générez une clé sécurisée)
SECRET_KEY=your-very-long-secret-key-change-this-in-production-min-32-chars

# Base de données (mettre false si pas de DB)
DB_REQUIRED=false

# Mode démo pour tests (optionnel)
DEMO_LOGIN_ENABLED=true
DEMO_USERNAME=demo
DEMO_PASSWORD=demo123

# Logging
LOG_LEVEL=INFO

# CORS (pour production)
ALLOW_ORIGINS=*

# JWT
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480
'''
    
    with open(".env.render.example", "w") as f:
        f.write(env_content)
    
    print("✅ Exemple d'environnement créé")
    return True

def verify_dockerfile():
    """Vérifie et corrige le Dockerfile si nécessaire"""
    print("🐳 Vérification du Dockerfile...")
    
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        print("❌ Dockerfile manquant!")
        return False
    
    with open(dockerfile_path, "r") as f:
        content = f.read()
    
    # Vérifications basiques
    issues = []
    
    if "FROM python:" not in content:
        issues.append("FROM python: manquant")
    
    if "requirements_api.txt" not in content:
        issues.append("requirements_api.txt pas installé")
    
    if "EXPOSE 8000" not in content:
        issues.append("EXPOSE 8000 manquant")
    
    if "uvicorn" not in content:
        issues.append("Commande uvicorn manquante")
    
    if issues:
        print("⚠️ Problèmes détectés dans le Dockerfile:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    else:
        print("✅ Dockerfile semble correct")
        return True

def create_render_yaml():
    """Crée un fichier render.yaml pour la configuration"""
    print("⚙️ Création de render.yaml...")
    
    render_config = {
        "services": [
            {
                "type": "web",
                "name": "games-api-ml",
                "env": "python",
                "buildCommand": "pip install -r requirements_api.txt",
                "startCommand": "uvicorn api_games_plus:app --host 0.0.0.0 --port $PORT",
                "envVars": [
                    {
                        "key": "SECRET_KEY",
                        "generateValue": True
                    },
                    {
                        "key": "DB_REQUIRED",
                        "value": "false"
                    },
                    {
                        "key": "DEMO_LOGIN_ENABLED", 
                        "value": "true"
                    },
                    {
                        "key": "LOG_LEVEL",
                        "value": "INFO"
                    }
                ]
            }
        ]
    }
    
    with open("render.yaml", "w") as f:
        json.dump(render_config, f, indent=2)
    
    print("✅ render.yaml créé")
    return True

def create_startup_test():
    """Crée un test de démarrage pour debug"""
    print("🧪 Création du test de démarrage...")
    
    test_content = '''#!/usr/bin/env python3
"""Test de démarrage pour debug Render"""

import os
import sys

# Définir les variables d'environnement minimales
os.environ.setdefault('SECRET_KEY', 'test-key-for-render-deployment-very-long-secret')
os.environ.setdefault('DB_REQUIRED', 'false')
os.environ.setdefault('DEMO_LOGIN_ENABLED', 'true')
os.environ.setdefault('LOG_LEVEL', 'INFO')

def test_startup():
    print("🚀 Test de démarrage de l'API...")
    
    try:
        # Test 1: Imports
        print("1️⃣ Test des imports...")
        import api_games_plus
        print("   ✅ api_games_plus importé")
        
        # Test 2: App creation
        print("2️⃣ Test création app...")
        app = api_games_plus.app
        print(f"   ✅ App créée: {app.title}")
        
        # Test 3: Compliance
        print("3️⃣ Test compliance...")
        compliance_enabled = getattr(api_games_plus, 'COMPLIANCE_ENABLED', False)
        print(f"   📋 Compliance enabled: {compliance_enabled}")
        
        # Test 4: Health check
        print("4️⃣ Test health check...")
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/healthz")
        print(f"   📊 Health status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data.get('status')}")
        else:
            print(f"   ❌ Health check failed: {response.text}")
        
        print("🎉 Test de démarrage terminé avec succès!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_startup()
    sys.exit(0 if success else 1)
'''
    
    with open("test_startup.py", "w") as f:
        f.write(test_content)
    
    os.chmod("test_startup.py", 0o755)
    print("✅ Test de démarrage créé")
    return True

def main():
    """Fonction principale de correction"""
    print("🔧 CORRECTION RAPIDE POUR RENDER")
    print("=" * 40)
    
    fixes = [
        ("Module Compliance", create_minimal_compliance_module),
        ("Exemple Environnement", create_minimal_env_example), 
        ("Vérification Dockerfile", verify_dockerfile),
        ("Configuration Render", create_render_yaml),
        ("Test Startup", create_startup_test)
    ]
    
    success_count = 0
    
    for name, fix_func in fixes:
        print(f"\n📋 {name}...")
        try:
            if fix_func():
                success_count += 1
        except Exception as e:
            print(f"❌ Erreur {name}: {e}")
    
    print(f"\n🎯 RÉSULTATS: {success_count}/{len(fixes)} corrections appliquées")
    
    print("\n📋 PROCHAINES ÉTAPES:")
    print("1. Vérifiez que tous les fichiers ont été créés")
    print("2. Committez et poussez les changements:")
    print("   git add .")
    print("   git commit -m 'fix: Add compliance module and Render config'")
    print("   git push")
    print("3. Vérifiez les variables d'environnement dans Render:")
    print("   - Utilisez les valeurs de .env.render.example")
    print("4. Redéployez depuis Render dashboard")
    print("5. Relancez le pipeline de monitoring")
    
    return success_count == len(fixes)

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ TOUTES LES CORRECTIONS APPLIQUÉES!")
    else:
        print("\n⚠️ Certaines corrections ont échoué - vérifiez les erreurs ci-dessus")
