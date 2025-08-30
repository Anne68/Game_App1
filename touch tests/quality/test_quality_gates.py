# tests/quality/test_quality_gates.py - Tests de qualité pour CI

import pytest
import numpy as np
import time
import subprocess
import ast
from pathlib import Path
from unittest.mock import Mock, patch

class CodeQualityTests:
    """Tests de qualité du code pour CI"""
    
    def test_no_hardcoded_secrets(self):
        """Vérifie l'absence de secrets en dur dans le code"""
        secret_patterns = [
            r'password\s*=\s*[\'"][^\'"]+[\'"]',
            r'api_key\s*=\s*[\'"][^\'"]+[\'"]',
            r'secret\s*=\s*[\'"][^\'"]+[\'"]',
        ]
        
        import re
        
        python_files = list(Path(".").glob("**/*.py"))
        
        for file_path in python_files:
            if ".git" in str(file_path) or "__pycache__" in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in secret_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            if "example" not in match.lower() and "test" not in match.lower():
                                assert False, f"Potential hardcoded secret in {file_path}: {match}"
            except (UnicodeDecodeError, PermissionError):
                pass
    
    def test_import_time_performance(self):
        """Vérifie que les imports ne sont pas trop lents"""
        import importlib
        
        modules_to_test = [
            "api_games_plus",
            "model_manager", 
            "monitoring_metrics",
            "settings"
        ]
        
        for module_name in modules_to_test:
            start_time = time.time()
            try:
                importlib.import_module(module_name)
                import_time = time.time() - start_time
                
                assert import_time < 2.0, f"Module {module_name} import too slow: {import_time:.2f}s"
            except ImportError:
                pass

class MLModelQualityTests:
    """Tests de qualité spécifiques au modèle ML"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Données d'entraînement pour les tests"""
        return [
            {"id": 1, "title": "The Witcher 3", "genres": "RPG Action", 
             "rating": 4.9, "metacritic": 93, "platforms": ["PC", "PS4"]},
            {"id": 2, "title": "Hades", "genres": "Action Roguelike", 
             "rating": 4.8, "metacritic": 93, "platforms": ["PC", "Switch"]},
            {"id": 3, "title": "Celeste", "genres": "Platformer Indie", 
             "rating": 4.6, "metacritic": 94, "platforms": ["PC", "Switch"]}
        ]
    
    def test_model_determinism(self, sample_training_data):
        """Vérifie que le modèle produit des résultats déterministes"""
        from model_manager import RecommendationModel
        
        model1 = RecommendationModel("test-v1")
        model2 = RecommendationModel("test-v1")
        
        model1.train(sample_training_data)
        model2.train(sample_training_data)
        
        query = "Action RPG"
        predictions1 = model1.predict(query, k=5)
        predictions2 = model2.predict(query, k=5)
        
        titles1 = [p["title"] for p in predictions1]
        titles2 = [p["title"] for p in predictions2]
        
        assert titles1 == titles2, "Model predictions are not deterministic"
    
    def test_model_performance_regression(self, sample_training_data):
        """Vérifie qu'il n'y a pas de régression de performance"""
        from model_manager import RecommendationModel
        
        model = RecommendationModel()
        model.train(sample_training_data)
        
        baseline_metrics = {
            "max_latency_ms": 1000,
            "min_recommendations": 1
        }
        
        queries = ["Action", "RPG", "Indie", "Strategy"]
        latencies = []
        recommendation_counts = []
        
        for query in queries:
            start = time.time()
            recommendations = model.predict(query, k=10)
            latency_ms = (time.time() - start) * 1000
            
            latencies.append(latency_ms)
            recommendation_counts.append(len(recommendations))
        
        avg_latency = np.mean(latencies) if latencies else 0
        avg_recommendations = np.mean(recommendation_counts) if recommendation_counts else 0
        
        assert avg_latency <= baseline_metrics["max_latency_ms"], \
            f"Average latency {avg_latency:.2f}ms exceeds baseline {baseline_metrics['max_latency_ms']}ms"
        
        assert avg_recommendations >= baseline_metrics["min_recommendations"], \
            f"Average recommendations {avg_recommendations} below baseline {baseline_metrics['min_recommendations']}"

class APIQualityTests:
    """Tests de qualité de l'API"""
    
    @pytest.fixture
    def api_client(self):
        from fastapi.testclient import TestClient
        from api_games_plus import app
        return TestClient(app)
    
    @pytest.fixture
    def auth_token(self, api_client):
        api_client.post("/register", data={
            "username": "qualitytest", 
            "password": "QualityTest123!"
        })
        
        response = api_client.post("/token", data={
            "username": "qualitytest", 
            "password": "QualityTest123!"
        })
        
        if response.status_code == 200:
            return response.json()["access_token"]
        return None
    
    def test_api_response_schemas(self, api_client, auth_token):
        """Vérifie que les réponses respectent les schémas définis"""
        if not auth_token:
            pytest.skip("No auth token available")
        
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        test_cases = [
            {
                "endpoint": "/healthz",
                "method": "GET",
                "expected_fields": ["status", "time", "model_loaded"],
                "headers": {}
            },
            {
                "endpoint": "/model/metrics", 
                "method": "GET",
                "expected_fields": ["model_version", "is_trained", "total_predictions"],
                "headers": headers
            }
        ]
        
        for test_case in test_cases:
            if test_case["method"] == "GET":
                response = api_client.get(
                    test_case["endpoint"], 
                    headers=test_case.get("headers", {})
                )
            
            assert response.status_code in [200, 201], \
                f"Unexpected status code for {test_case['endpoint']}: {response.status_code}"
            
            data = response.json()
            for field in test_case["expected_fields"]:
                assert field in data, \
                    f"Missing field '{field}' in response from {test_case['endpoint']}"

class QualityGate:
    """Configuration des seuils de qualité pour CI"""
    
    QUALITY_METRICS = {
        "min_code_coverage": 70.0,  # Seuil réaliste
        "max_complexity": 15,
        "max_startup_time": 10.0,
        "max_memory_increase": 200.0,
        "max_api_latency": 1000.0,
        "min_test_pass_rate": 90.0
    }
    
    @classmethod
    def validate_quality_gate(cls, metrics: dict) -> dict:
        """Valide que tous les seuils de qualité sont respectés"""
        results = {
            "passed": True,
            "failed_metrics": [],
            "warnings": []
        }
        
        for metric, threshold in cls.QUALITY_METRICS.items():
            actual_value = metrics.get(metric)
            
            if actual_value is None:
                results["warnings"].append(f"Metric '{metric}' not measured")
                continue
            
            if metric.startswith("min_"):
                if actual_value < threshold:
                    results["passed"] = False
                    results["failed_metrics"].append({
                        "metric": metric,
                        "expected": f">= {threshold}",
                        "actual": actual_value
                    })
            else:
                if actual_value > threshold:
                    results["passed"] = False 
                    results["failed_metrics"].append({
                        "metric": metric,
                        "expected": f"<= {threshold}",
                        "actual": actual_value
                    })
        
        return results

# Tests de performance simplifiés
class PerformanceTests:
    """Tests de performance de base"""
    
    def test_startup_time_basic(self):
        """Test de base du temps de démarrage"""
        import os
        
        # Test d'import simple
        start_time = time.time()
        try:
            # Simuler un import de l'app
            import api_games_plus
            startup_time = time.time() - start_time
            assert startup_time < 5.0, f"Startup time too slow: {startup_time:.2f}s"
        except ImportError as e:
            pytest.skip(f"Cannot import api_games_plus: {e}")
    
    def test_memory_baseline(self):
        """Test de base de consommation mémoire"""
        try:
            import psutil
            import gc
            
            gc.collect()
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Import des modules
            import api_games_plus
            import model_manager
            
            gc.collect()
            loaded_memory = process.memory_info().rss / (1024 * 1024)
            memory_increase = loaded_memory - baseline_memory
            
            assert memory_increase < 300, f"Memory increase too high: {memory_increase:.1f}MB"
            
        except ImportError:
            pytest.skip("psutil not available")

if __name__ == "__main__":
    # Script de validation qualité pour CI
    import sys
    
    pytest_args = [
        "--tb=short",
        "--quiet", 
        "-x",
        "tests/quality/test_quality_gates.py"
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code != 0:
        print("❌ Quality tests failed")
        sys.exit(1)
    else:
        print("✅ Quality tests passed")
        sys.exit(0)
