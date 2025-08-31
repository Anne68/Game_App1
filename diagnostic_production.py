#!/usr/bin/env python3
"""
Diagnostic de l'API en production - Troubleshooting health check
"""

import requests
import json
import time
from datetime import datetime

def diagnostic_api_production():
    """Diagnostic complet de l'API en production"""
    
    print("🔍 DIAGNOSTIC API PRODUCTION")
    print("=" * 50)
    
    api_url = "https://game-app-y8be.onrender.com"
    
    print(f"🔗 URL testée: {api_url}")
    print(f"⏰ Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Test 1: Connectivité de base
    print("1️⃣ Test de connectivité de base...")
    try:
        response = requests.get(api_url, timeout=30)
        print(f"   Status: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        if response.status_code == 200:
            print("   ✅ Connectivité OK")
        else:
            print(f"   ⚠️ Status inattendu: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Erreur connectivité: {e}")
        return False
    
    print()
    
    # Test 2: Health endpoint détaillé
    print("2️⃣ Test health endpoint détaillé...")
    try:
        start_time = time.time()
        response = requests.get(f"{api_url}/healthz", timeout=30)
        response_time = time.time() - start_time
        
        print(f"   Status: {response.status_code}")
        print(f"   Temps de réponse: {response_time:.2f}s")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("   ✅ Health endpoint OK")
                print("   📊 Détails:")
                for key, value in data.items():
                    print(f"      {key}: {value}")
            except json.JSONDecodeError:
                print("   ⚠️ Réponse non-JSON:")
                print(f"   {response.text[:500]}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            print("   📄 Response headers:")
            for key, value in response.headers.items():
                print(f"      {key}: {value}")
            print("   📄 Response body:")
            print(f"   {response.text[:1000]}")
            
            # Analyser le type d'erreur
            if response.status_code == 500:
                print("   🔍 Erreur 500 - Problème serveur interne")
                print("   💡 Causes possibles:")
                print("      - Erreur dans le code Python")
                print("      - Problème de base de données")
                print("      - Module manquant")
                print("      - Variable d'environnement manquante")
            elif response.status_code == 502:
                print("   🔍 Erreur 502 - Bad Gateway")
                print("   💡 L'application ne démarre probablement pas")
            elif response.status_code == 503:
                print("   🔍 Erreur 503 - Service Unavailable")
                print("   💡 L'application est temporairement indisponible")
                
    except requests.exceptions.Timeout:
        print("   ❌ Timeout - L'API met trop de temps à répondre")
    except requests.exceptions.ConnectionError:
        print("   ❌ Erreur de connexion - API inaccessible")
    except Exception as e:
        print(f"   ❌ Erreur inattendue: {e}")
    
    print()
    
    # Test 3: Autres endpoints
    print("3️⃣ Test autres endpoints...")
    test_endpoints = [
        "/",
        "/docs",
        "/metrics"
    ]
    
    for endpoint in test_endpoints:
        try:
            response = requests.get(f"{api_url}{endpoint}", timeout=10)
            status_icon = "✅" if response.status_code < 400 else "❌"
            print(f"   {status_icon} {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ {endpoint}: Error - {e}")
    
    print()
    
    # Test 4: Diagnostic des logs Render
    print("4️⃣ Recommandations de diagnostic...")
    print("   🔧 Actions à effectuer:")
    print("   1. Vérifier les logs Render:")
    print("      - Aller sur https://dashboard.render.com")
    print("      - Sélectionner votre service")
    print("      - Cliquer sur 'Logs' pour voir les erreurs")
    print()
    print("   2. Vérifier les variables d'environnement:")
    print("      - SECRET_KEY définie")
    print("      - DB_REQUIRED=false (si pas de DB)")
    print("      - DEMO_LOGIN_ENABLED=true")
    print()
    print("   3. Vérifier le déploiement:")
    print("      - Dernière version déployée")
    print("      - Processus de build réussi")
    print("      - Dockerfile correct")
    print()
    
    return False

def test_local_version():
    """Test de la version locale pour comparaison"""
    print("5️⃣ Test version locale (si disponible)...")
    try:
        response = requests.get("http://localhost:8000/healthz", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Version locale fonctionne")
            print(f"   📊 Status local: {data.get('status')}")
            return True
        else:
            print(f"   ❌ Version locale: {response.status_code}")
    except Exception:
        print("   ⚠️ Version locale non accessible (normal si pas démarrée)")
    return False

def generate_fix_suggestions():
    """Génère des suggestions de correction"""
    print()
    print("🔧 SUGGESTIONS DE CORRECTION")
    print("=" * 30)
    
    suggestions = [
        {
            "problem": "Erreur 500 - Internal Server Error",
            "solutions": [
                "Vérifier les logs Render pour l'erreur exacte",
                "S'assurer que tous les modules sont installés (requirements_api.txt)",
                "Vérifier que les variables d'environnement sont définies",
                "Tester le code localement avant redéploiement"
            ]
        },
        {
            "problem": "Module compliance manquant",
            "solutions": [
                "Créer le dossier compliance/ avec __init__.py",
                "Ajouter standards_compliance.py avec les classes",
                "Redéployer l'application"
            ]
        },
        {
            "problem": "Variable d'environnement manquante",
            "solutions": [
                "Ajouter SECRET_KEY dans les settings Render",
                "Définir DB_REQUIRED=false si pas de base de données",
                "Ajouter DEMO_LOGIN_ENABLED=true pour le mode démo"
            ]
        }
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion['problem']}")
        for solution in suggestion['solutions']:
            print(f"   • {solution}")
        print()

if __name__ == "__main__":
    success = diagnostic_api_production()
    
    if not success:
        test_local_version()
        generate_fix_suggestions()
    
    print("🎯 NEXT STEPS:")
    print("1. Consulter les logs Render pour l'erreur exacte")
    print("2. Appliquer les corrections identifiées")
    print("3. Redéployer et retester")
    print("4. Relancer le pipeline de monitoring")
