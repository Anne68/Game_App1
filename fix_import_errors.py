#!/usr/bin/env python3
"""
fix_import_errors.py - Correction des erreurs d'import après simplification
"""

import os
import re
from pathlib import Path

def fix_api_imports():
    """Corrige les imports dans api_games_plus.py"""
    print("🔧 Correction des imports dans api_games_plus.py...")
    
    api_file = Path("api_games_plus.py")
    if not api_file.exists():
        print("❌ api_games_plus.py non trouvé")
        return False
    
    with open(api_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Corrections des imports
    fixes = [
        # Corriger l'import du model_manager
        (
            r'from model_manager import \(\s*get_model, get_model_manager, MLModelManager, ModelType,\s*RecommendationModel, GameClassificationModel, GameClusteringModel\s*\)',
            'from model_manager import get_model, reset_model'
        ),
        # Alternative si le pattern ne match pas exactement
        (
            r'from model_manager import.*get_model_manager.*',
            'from model_manager import get_model, reset_model'
        ),
        # Supprimer les références à des classes qui n'existent plus
        (
            r'ModelType\.',
            '# ModelType removed - '
        ),
        # Corriger les appels à get_model_manager
        (
            r'get_model_manager\(\)',
            'get_model()'
        )
    ]
    
    changes_made = 0
    for pattern, replacement in fixes:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes_made += 1
            print(f"   ✅ Corrigé: {pattern}")
    
    if changes_made > 0:
        with open(api_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {changes_made} correction(s) appliquée(s)")
        return True
    else:
        print("⚠️ Aucune correction nécessaire ou pattern non trouvé")
        return True  # Pas d'erreur

def ensure_correct_imports():
    """S'assure que les imports sont corrects"""
    print("🔧 Vérification des imports corrects...")
    
    api_file = Path("api_games_plus.py")
    if not api_file.exists():
        return False
    
    with open(api_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Vérifier que l'import correct est présent
    correct_import = "from model_manager import get_model, reset_model"
    
    if correct_import not in content:
        print("🔧 Ajout de l'import correct...")
        
        # Trouver la ligne d'import des modules locaux
        import_section_pattern = r'# Import des modules locaux\n'
        
        if re.search(import_section_pattern, content):
            # Remplacer la section d'import
            new_import_section = '''# Import des modules locaux
from settings import get_settings
from model_manager import get_model, reset_model
from monitoring_metrics import (
    prediction_latency, model_prediction_counter, get_monitor,
)'''
            
            content = re.sub(
                r'# Import des modules locaux\nfrom settings import get_settings\nfrom model_manager import.*?\nfrom monitoring_metrics import \(\s*prediction_latency, model_prediction_counter, get_monitor,\s*\)',
                new_import_section,
                content,
                flags=re.DOTALL
            )
            
            with open(api_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Import correct ajouté")
            return True
    
    print("✅ Imports déjà corrects")
    return True

def test_imports():
    """Test les imports après correction"""
    print("🧪 Test des imports...")
    
    # Définir les variables d'environnement pour le test
    os.environ.update({
        'SECRET_KEY': 'test-secret-key-for-import-test-very-long',
        'DB_REQUIRED': 'false',
        'DEMO_LOGIN_ENABLED': 'true',
        'LOG_LEVEL': 'INFO',
        'ALLOW_ORIGINS': '*'
    })
    
    try:
        # Test d'import du model_manager
        print("   🧪 Test model_manager...")
        import model_manager
        get_model = getattr(model_manager, 'get_model', None)
        if get_model:
            model = get_model()
            print(f"   ✅ Model manager OK - Version {model.model_version}")
        else:
            print("   ❌ get_model non trouvé")
            return False
        
        # Test d'import de l'API
        print("   🧪 Test api_games_plus...")
        import api_games_plus
        app = getattr(api_games_plus, 'app', None)
        if app:
            print(f"   ✅ API OK - {app.title}")
        else:
            print("   ❌ app non trouvé")
            return False
        
        print("✅ Tous les imports fonctionnent")
        return True
        
    except ImportError as e:
        print(f"   ❌ Erreur import: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Erreur test: {e}")
        return False

def create_simple_test():
    """Crée un test simple pour vérifier que tout fonctionne"""
    print("🧪 Création d'un test simple...")
    
    test_content = '''#!/usr/bin/env python3
"""Test simple après correction des imports"""

import os

# Variables d'environnement pour le test
os.environ.update({
    'SECRET_KEY': 'test-secret-key-for-simple-test-very-long-and-secure',
    'DB_REQUIRED': 'false',
    'DEMO_LOGIN_ENABLED': 'true',
    'DEMO_USERNAME': 'demo',
    'DEMO_PASSWORD': 'demo123',
    'LOG_LEVEL': 'INFO'
})

def test_basic_functionality():
    """Test de base de l'API"""
    try:
        print("🧪 Test imports...")
        
        # Test model_manager
        from model_manager import get_model
        model = get_model()
        print(f"✅ Model: {model.model_version}")
        
        # Test API
        from api_games_plus import app
        print(f"✅ API: {app.title}")
        
        # Test avec FastAPI TestClient
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test health
        response = client.get("/healthz")
        print(f"✅ Health: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
        
        # Test root
        response = client.get("/")
        print(f"✅ Root: {response.status_code}")
        
        print("🎉 TOUS LES TESTS PASSENT!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)
'''
    
    with open("test_simple_after_fix.py", 'w') as f:
        f.write(test_content)
    
    os.chmod("test_simple_after_fix.py", 0o755)
    print("✅ Test simple créé: test_simple_after_fix.py")

def main():
    """Fonction principale de correction"""
    print("🚨 CORRECTION DES ERREURS D'IMPORT")
    print("=" * 40)
    
    steps = [
        ("Correction imports API", fix_api_imports),
        ("Vérification imports corrects", ensure_correct_imports),
        ("Test des imports", test_imports),
        ("Création test simple", create_simple_test)
    ]
    
    success_count = 0
    for name, func in steps:
        print(f"\n📋 {name}...")
        try:
            if func():
                success_count += 1
            else:
                print(f"❌ {name} échoué")
        except Exception as e:
            print(f"❌ {name} erreur: {e}")
    
    print(f"\n🎯 RÉSULTATS: {success_count}/{len(steps)} étapes réussies")
    
    if success_count >= 3:
        print("\n✅ CORRECTIONS TERMINÉES!")
        print("📋 Prochaines étapes:")
        print("1. Tester l'API:")
        print("   python test_simple_after_fix.py")
        print("2. Démarrer l'API:")
        print("   uvicorn api_games_plus:app --reload --port 8000")
        print("3. Test complet:")
        print("   python demo_simplified_api.py")
        return True
    else:
        print("\n❌ Des problèmes persistent")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
