#!/usr/bin/env python3
"""Test rapide pour vérifier que l'app fonctionne"""

import os

# Variables d'environnement pour test
os.environ.update({
    'SECRET_KEY': 'test-key-very-long-for-security-requirements',
    'DB_REQUIRED': 'false',
    'DEMO_LOGIN_ENABLED': 'true',
    'DEMO_USERNAME': 'demo',
    'DEMO_PASSWORD': 'demo123',
    'LOG_LEVEL': 'INFO',
    'ALLOW_ORIGINS': '*'
})

def test_imports():
    """Test des imports critiques"""
    try:
        print("🔍 Testing imports...")
        
        # Test imports de base
        import fastapi
        import uvicorn
        import pymysql
        import sklearn
        print("  ✅ Basic dependencies")
        
        # Test modules application
        import settings
        import model_manager
        import monitoring_metrics
        print("  ✅ Application modules")
        
        # Test import principal
        import api_games_plus
        print("  ✅ Main API module")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_app_creation():
    """Test création de l'application FastAPI"""
    try:
        print("🚀 Testing FastAPI app creation...")
        
        from api_games_plus import app
        print(f"  ✅ App created: {app.title} v{app.version}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ App creation error: {e}")
        return False

def test_health_endpoint():
    """Test de l'endpoint de santé"""
    try:
        print("🏥 Testing health endpoint...")
        
        from fastapi.testclient import TestClient
        from api_games_plus import app
        
        client = TestClient(app)
        response = client.get("/healthz")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Health check: {data.get('status')}")
            print(f"  📊 Model loaded: {data.get('model_loaded')}")
            print(f"  🗄️ DB ready: {data.get('db_ready')}")
            return True
        else:
            print(f"  ❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Health endpoint error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST RAPIDE - Games API ML")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("App Creation", test_app_creation),
        ("Health Endpoint", test_health_endpoint)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n📋 {name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {name} FAILED")
    
    print("\n" + "=" * 60)
    if passed == total:
        print("🎉 TOUS LES TESTS PASSÉS!")
        print("✅ L'application est prête pour le déploiement")
    else:
        print(f"❌ {total - passed}/{total} tests échoués")
        print("🔧 Vérifiez les erreurs ci-dessus")
    print("=" * 60)
