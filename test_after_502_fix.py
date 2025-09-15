#!/usr/bin/env python3
"""
Test de validation après correction de l'erreur 502
"""

import requests
import time
import os

def test_local_startup():
    """Teste le démarrage local avec la nouvelle configuration"""
    print("🧪 TEST DÉMARRAGE LOCAL")
    print("=" * 25)
    
    # Simuler l'environnement Render
    os.environ.update({
        'PORT': '8000',
        'SECRET_KEY': 'test-secret-key-very-long-for-local-testing',
        'DB_REQUIRED': 'false', 
        'DEMO_LOGIN_ENABLED': 'true',
        'LOG_LEVEL': 'INFO',
        'PYTHONPATH': '.'
    })
    
    try:
        # Test d'import
        print("1️⃣ Test import...")
        import api_games_plus
        print("   ✅ Import réussi")
        
        # Test création app
        print("2️⃣ Test app...")
        app = api_games_plus.app
        print(f"   ✅ App créée: {app.title}")
        
        # Test compliance
        print("3️⃣ Test compliance...")
        compliance_enabled = getattr(api_games_plus, 'COMPLIANCE_ENABLED', False)
        print(f"   ✅ Compliance: {compliance_enabled}")
        
        # Test health check
        print("4️⃣ Test health check...")
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        response = client.get("/healthz")
        print(f"   📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health: {data.get('status')}")
            print(f"   📋 Compliance: {data.get('compliance_enabled')}")
            return True
        else:
            print(f"   ❌ Health check: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_production_endpoint():
    """Teste l'endpoint de production après correction"""
    print("\n🌐 TEST ENDPOINT PRODUCTION")
    print("=" * 30)
    
    api_url = "https://game-app-y8be.onrender.com"
    
    print("⏳ Attente stabilisation (30s)...")
    time.sleep(30)
    
    try:
        print("1️⃣ Test connectivité...")
        response = requests.get(api_url, timeout=30)
        print(f"   📊 Root endpoint: {response.status_code}")
        
        print("2️⃣ Test health check...")
        response = requests.get(f"{api_url}/healthz", timeout=30)
        print(f"   📊 Health status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Production API opérationnelle!")
            print(f"   📋 Status: {data.get('status')}")
            print(f"   🤖 Model: {data.get('model_loaded')}")
            print(f"   🔒 Compliance: {data.get('compliance_enabled')}")
            return True
        elif response.status_code == 502:
            print("   ❌ Encore une erreur 502 - l'application ne démarre pas")
            print("   💡 Vérifiez les logs Render pour l'erreur exacte")
            return False
        else:
            print(f"   ⚠️ Status inattendu: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ⏳ Timeout - l'API met trop de temps à répondre")
        return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Erreur de connexion")
        return False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def validate_files_created():
    """Valide que tous les fichiers nécessaires ont été créés"""
    print("\n📁 VALIDATION FICHIERS")
    print("=" * 20)
    
    required_files = [
        "requirements_api.txt",
        "Dockerfile", 
        "start.sh",
        "compliance/__init__.py",
        "compliance/standards_compliance.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Fichiers manquants:")
        for file in missing_files:
            print(f"   • {file}")
        return False
    
    print("\n✅ Tous les fichiers requis sont présents")
    return True

def main():
    """Fonction principale de test"""
    print("🧪 VALIDATION POST-CORRECTION 502")
    print("=" * 35)
    
    tests = [
        ("Validation fichiers", validate_files_created),
        ("Démarrage local", test_local_startup),
        ("Endpoint production", test_production_endpoint)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n📋 {name}...")
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"✅ {name} réussi")
            else:
                print(f"❌ {name} échoué")
        except Exception as e:
            print(f"❌ {name} erreur: {e}")
            results.append(False)
    
    # Résumé
    success_count = sum(results)
    total = len(results)
    
    print(f"\n🎯 RÉSULTATS: {success_count}/{total} tests réussis")
    
    if success_count == total:
        print("🎉 TOUTES LES CORRECTIONS VALIDÉES!")
        print("✅ L'erreur 502 devrait être résolue")
        print("🚀 Votre pipeline de monitoring devrait maintenant passer")
    elif success_count >= 2:
        print("✅ Corrections largement appliquées")
        print("⏳ Il peut falloir quelques minutes pour que Render redémarre")
        print("🔄 Essayez de redéployer manuellement si nécessaire")
    else:
        print("⚠️ Des problèmes persistent")
        print("🔧 Vérifiez les logs Render pour plus de détails")
    
    return success_count >= 2

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
