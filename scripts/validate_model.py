#!/usr/bin/env python3
"""
scripts/validate_model.py - Validation automatique du modèle en production
"""

import sys
import requests
import json
import time
from typing import Dict, List, Optional
import argparse
from dataclasses import dataclass

@dataclass
class ValidationResult:
    success: bool
    message: str
    details: Optional[Dict] = None

class ModelValidator:
    """Validateur de modèle ML pour le CI/CD"""
    
    def __init__(self, api_url: str, username: str = None, password: str = None):
        self.api_url = api_url.rstrip('/')
        self.username = username
        self.password = password
        self.token = None
        self.session = requests.Session()
        self.session.timeout = 30
    
    def authenticate(self) -> ValidationResult:
        """Authentification auprès de l'API"""
        if not self.username or not self.password:
            return ValidationResult(False, "Identifiants manquants")
        
        try:
            response = self.session.post(
                f"{self.api_url}/token",
                data={"username": self.username, "password": self.password}
            )
            
            if response.status_code == 200:
                self.token = response.json()["access_token"]
                self.session.headers.update({
                    "Authorization": f"Bearer {self.token}"
                })
                return ValidationResult(True, "Authentification réussie")
            else:
                return ValidationResult(
                    False, 
                    f"Échec authentification: {response.status_code}",
                    {"response": response.text}
                )
        
        except Exception as e:
            return ValidationResult(False, f"Erreur authentification: {e}")
    
    def check_api_health(self) -> ValidationResult:
        """Vérification de santé de l'API"""
        try:
            response = self.session.get(f"{self.api_url}/healthz")
            
            if response.status_code == 200:
                health_data = response.json()
                status = health_data.get("status", "unknown")
                model_loaded = health_data.get("model_loaded", False)
                
                if status == "healthy" and model_loaded:
                    return ValidationResult(
                        True, 
                        "API en bonne santé",
                        health_data
                    )
                else:
                    return ValidationResult(
                        False,
                        f"API dégradée: status={status}, model_loaded={model_loaded}",
                        health_data
                    )
            else:
                return ValidationResult(
                    False,
                    f"API non accessible: {response.status_code}"
                )
        
        except Exception as e:
            return ValidationResult(False, f"Erreur vérification santé: {e}")
    
    def validate_model_training(self) -> ValidationResult:
        """Validation de l'entraînement du modèle"""
        try:
            response = self.session.get(f"{self.api_url}/model/metrics")
            
            if response.status_code == 200:
                metrics = response.json()
                is_trained = metrics.get("is_trained", False)
                games_count = metrics.get("games_count", 0)
                model_version = metrics.get("model_version", "unknown")
                
                if is_trained and games_count > 0:
                    return ValidationResult(
                        True,
                        f"Modèle entraîné: v{model_version}, {games_count} jeux",
                        metrics
                    )
                else:
                    return ValidationResult(
                        False,
                        f"Modèle non entraîné: trained={is_trained}, games={games_count}",
                        metrics
                    )
            else:
                return ValidationResult(
                    False,
                    f"Impossible d'obtenir les métriques: {response.status_code}"
                )
        
        except Exception as e:
            return ValidationResult(False, f"Erreur validation modèle: {e}")
    
    def test_predictions(self) -> ValidationResult:
        """Test des prédictions ML"""
        test_queries = [
            {"query": "Action RPG", "expected_min": 1},
            {"query": "Indie", "expected_min": 0},  # Peut ne pas avoir de résultats
            {"query": "Strategy", "expected_min": 0},
            {"query": "Simulation", "expected_min": 0}
        ]
        
        results = []
        failures = []
        
        for test in test_queries:
            try:
                response = self.session.post(
                    f"{self.api_url}/recommend/ml",
                    json={"query": test["query"], "k": 5, "min_confidence": 0.0}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])
                    latency = data.get("latency_ms", 0)
                    
                    if len(recommendations) >= test["expected_min"]:
                        results.append({
                            "query": test["query"],
                            "count": len(recommendations),
                            "latency_ms": latency,
                            "status": "success"
                        })
                    else:
                        failures.append(
                            f"Query '{test['query']}': {len(recommendations)} résultats "
                            f"(attendu >= {test['expected_min']})"
                        )
                else:
                    failures.append(
                        f"Query '{test['query']}' échouée: {response.status_code}"
                    )
            
            except Exception as e:
                failures.append(f"Query '{test['query']}' erreur: {e}")
        
        if failures:
            return ValidationResult(
                False,
                f"Tests prédictions échoués: {'; '.join(failures)}",
                {"results": results, "failures": failures}
            )
        else:
            avg_latency = sum(r["latency_ms"] for r in results) / len(results)
            return ValidationResult(
                True,
                f"Tests prédictions réussis (latence moy: {avg_latency:.1f}ms)",
                {"results": results, "avg_latency_ms": avg_latency}
            )
    
    def check_model_drift(self) -> ValidationResult:
        """Vérification du drift du modèle"""
        try:
            response = self.session.get(f"{self.api_url}/monitoring/drift")
            
            if response.status_code == 200:
                drift_data = response.json()
                status = drift_data.get("status", "unknown")
                drift_score = drift_data.get("drift_score", 0)
                
                if status == "high_drift":
                    return ValidationResult(
                        False,
                        f"DRIFT ÉLEVÉ détecté: {drift_score:.3f}",
                        drift_data
                    )
                elif status == "moderate_drift":
                    return ValidationResult(
                        True,
                        f"Drift modéré détecté: {drift_score:.3f} (surveiller)",
                        drift_data
                    )
                else:
                    return ValidationResult(
                        True,
                        f"Drift acceptable: {drift_score:.3f}",
                        drift_data
                    )
            else:
                return ValidationResult(
                    True,  # Non critique si endpoint indisponible
                    f"Vérification drift non disponible: {response.status_code}"
                )
        
        except Exception as e:
            return ValidationResult(True, f"Erreur vérification drift: {e}")
    
    def validate_performance(self) -> ValidationResult:
        """Test de performance basique"""
        try:
            start_time = time.time()
            
            # Série de prédictions rapides
            for i in range(5):
                response = self.session.post(
                    f"{self.api_url}/recommend/ml",
                    json={"query": f"Action {i}", "k": 3}
                )
                if response.status_code != 200:
                    return ValidationResult(
                        False,
                        f"Prédiction {i} échouée: {response.status_code}"
                    )
            
            total_time = time.time() - start_time
            avg_time_per_request = (total_time / 5) * 1000  # ms
            
            # SLA: moins de 1 seconde par prédiction en moyenne
            if avg_time_per_request < 1000:
                return ValidationResult(
                    True,
                    f"Performance acceptable: {avg_time_per_request:.1f}ms/req",
                    {"avg_time_ms": avg_time_per_request, "total_requests": 5}
                )
            else:
                return ValidationResult(
                    False,
                    f"Performance dégradée: {avg_time_per_request:.1f}ms/req",
                    {"avg_time_ms": avg_time_per_request, "total_requests": 5}
                )
        
        except Exception as e:
            return ValidationResult(False, f"Erreur test performance: {e}")
    
    def run_full_validation(self) -> bool:
        """Exécution de la validation complète"""
        print("🤖 Validation du modèle ML en cours...")
        print(f"🔗 API: {self.api_url}")
        
        validations = [
            ("Santé API", self.check_api_health),
            ("Authentification", self.authenticate),
            ("Modèle entraîné", self.validate_model_training),
            ("Prédictions", self.test_predictions),
            ("Performance", self.validate_performance),
            ("Drift", self.check_model_drift)
        ]
        
        all_passed = True
        results = []
        
        for name, validation_func in validations:
            print(f"\n📋 {name}...")
            result = validation_func()
            results.append((name, result))
            
            if result.success:
                print(f"✅ {result.message}")
                if result.details:
                    # Afficher quelques détails importants
                    if "latency_ms" in str(result.details):
                        print(f"   📊 Détails: {json.dumps(result.details, indent=2)}")
            else:
                print(f"❌ {result.message}")
                if result.details:
                    print(f"   📋 Détails: {json.dumps(result.details, indent=2)}")
                all_passed = False
        
        # Résumé final
        print("\n" + "="*50)
        if all_passed:
            print("🎉 VALIDATION RÉUSSIE - Modèle prêt pour la production")
        else:
            print("💥 VALIDATION ÉCHOUÉE - Modèle non prêt")
            
            # Afficher les échecs
            failures = [name for name, result in results if not result.success]
            print(f"❌ Échecs: {', '.join(failures)}")
        
        return all_passed

def main():
    parser = argparse.ArgumentParser(description="Validation du modèle ML")
    parser.add_argument("--api-url", default="http://localhost:8000",
                       help="URL de l'API (défaut: http://localhost:8000)")
    parser.add_argument("--username", help="Nom d'utilisateur pour l'auth")
    parser.add_argument("--password", help="Mot de passe pour l'auth")
    parser.add_argument("--prod-url", help="URL de production (surcharge --api-url)")
    
    args = parser.parse_args()
    
    # Utiliser l'
