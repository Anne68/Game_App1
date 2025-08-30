# test_api.py - Tests pour l'API avec intégration ML
import pytest
from fastapi.testclient import TestClient
from api_games_plus import app
import json

client = TestClient(app)

class TestAPIEndpoints:
    """Tests des endpoints de l'API"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup pour chaque test"""
        # S'assurer qu'un utilisateur test existe
        client.post("/register", data={"username": "testuser", "password": "testpass123"})
        
        # Obtenir un token
        response = client.post("/token", data={"username": "testuser", "password": "testpass123"})
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {}
    
    def test_health_endpoint(self):
        """Test le endpoint de santé"""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
    
    def test_metrics_endpoint(self):
        """Test le endpoint Prometheus"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "model_predictions_total" in response.text
    
    def test_authentication_required(self):
        """Test que l'authentification est requise"""
        response = client.get("/games/by-title/test")
        assert response.status_code == 401
    
    def test_register_and_login(self):
        """Test l'inscription et la connexion"""
        # Register
        response = client.post("/register", 
                              data={"username": "newuser", "password": "newpass123"})
        assert response.status_code == 200 or response.status_code == 400  # 400 si déjà existe
        
        # Login
        response = client.post("/token", 
                              data={"username": "testuser", "password": "testpass123"})
        assert response.status_code == 200
        assert "access_token" in response.json()
    
    def test_search_games(self):
        """Test la recherche de jeux"""
        response = client.get("/games/by-title/witcher", headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) > 0
    
    def test_train_model(self):
        """Test l'entraînement du modèle"""
        response = client.post("/model/train", 
                              json={"force_retrain": True},
                              headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["success", "already_trained"]
        if data["status"] == "success":
            assert "duration" in data
            assert "version" in data
    
    def test_model_metrics(self):
        """Test les métriques du modèle"""
        # D'abord entraîner le modèle
        client.post("/model/train", json={"force_retrain": False}, headers=self.headers)
        
        response = client.get("/model/metrics", headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert "is_trained" in data
        assert "total_predictions" in data
    
    def test_ml_recommendations(self):
        """Test les recommandations ML"""
        # Entraîner d'abord si nécessaire
        client.post("/model/train", json={}, headers=self.headers)
        
        response = client.post("/recommend/ml",
                              json={"query": "Action RPG", "k": 5},
                              headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "model_version" in data
        assert len(data["recommendations"]) <= 5
        
        # Vérifier la structure des recommandations
        if data["recommendations"]:
            rec = data["recommendations"][0]
            assert "title" in rec
            assert "confidence" in rec
            assert 0 <= rec["confidence"] <= 1
    
    def test_recommendations_by_game(self):
        """Test les recommandations par jeu"""
        # Entraîner d'abord
        client.post("/model/train", json={}, headers=self.headers)
        
        response = client.get("/recommend/by-game/1?k=3", headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) <= 3
    
    def test_invalid_game_id(self):
        """Test avec un ID de jeu invalide"""
        client.post("/model/train", json={}, headers=self.headers)
        
        response = client.get("/recommend/by-game/999", headers=self.headers)
        assert response.status_code == 404
    
    def test_model_evaluation(self):
        """Test l'évaluation du modèle"""
        # Entraîner d'abord
        client.post("/model/train", json={}, headers=self.headers)
        
        response = client.post("/model/evaluate",
                              params={"test_queries": ["RPG", "Action"]},
                              headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "test_queries" in data
        assert "successes" in data
        assert "avg_confidence" in data
    
    def test_monitoring_summary(self):
        """Test le résumé du monitoring"""
        response = client.get("/monitoring/summary", headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "monitoring" in data
    
    def test_drift_detection(self):
        """Test la détection de drift"""
        response = client.get("/monitoring/drift", headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "drift_score" in data


class TestAPIPerformance:
    """Tests de performance de l'API"""
    
    @pytest.fixture
    def auth_headers(self):
        """Fixture pour l'authentification"""
        client.post("/register", data={"username": "perfuser", "password": "perfpass123"})
        response = client.post("/token", data={"username": "perfuser", "password": "perfpass123"})
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_concurrent_predictions(self, auth_headers):
        """Test des prédictions concurrentes"""
        import concurrent.futures
        
        # Entraîner le modèle
        client.post("/model/train", json={}, headers=auth_headers)
        
        def make_prediction():
            return client.post("/recommend/ml",
                             json={"query": "Action", "k": 5},
                             headers=auth_headers)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_prediction) for _ in range(10)]
            results = [f.result() for f in futures]
        
        # Toutes les requêtes devraient réussir
        assert all(r.status_code == 200 for r in results)
    
    def test_large_k_value(self, auth_headers):
        """Test avec une grande valeur de k"""
        client.post("/model/train", json={}, headers=auth_headers)
        
        response = client.post("/recommend/ml",
                              json={"query": "RPG", "k": 50},
                              headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        # Ne devrait pas retourner plus de jeux que disponibles
        assert len(data["recommendations"]) <= 7  # Nombre de jeux dans GAMES
    
    def test_response_time(self, auth_headers):
        """Test que les réponses sont rapides"""
        import time
        
        client.post("/model/train", json={}, headers=auth_headers)
        
        start = time.time()
        response = client.post("/recommend/ml",
                              json={"query": "Action RPG", "k": 5},
                              headers=auth_headers)
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 1.0  # Devrait répondre en moins d'1 seconde


class TestModelIntegration:
    """Tests d'intégration du modèle"""
    
    @pytest.fixture
    def auth_headers(self):
        client.post("/register", data={"username": "mluser", "password": "mlpass123"})
        response = client.post("/token", data={"username": "mluser", "password": "mlpass123"})
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_model_persistence(self, auth_headers, tmp_path):
        """Test que le modèle persiste après entraînement"""
        # Entraîner
        response = client.post("/model/train",
                              json={"version": "test-persist"},
                              headers=auth_headers)
        assert response.status_code == 200
        
        # Vérifier les métriques
        response = client.get("/model/metrics", headers=auth_headers)
        data = response.json()
        assert data["model_version"] == "test-persist"
        assert data["is_trained"] is True
    
    def test_recommendation_consistency(self, auth_headers):
        """Test que les recommandations sont consistantes"""
        client.post("/model/train", json={}, headers=auth_headers)
        
        # Faire deux requêtes identiques
        response1 = client.post("/recommend/ml",
                               json={"query": "RPG Action", "k": 3},
                               headers=auth_headers)
        response2 = client.post("/recommend/ml",
                               json={"query": "RPG Action", "k": 3},
                               headers=auth_headers)
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Les recommandations devraient être identiques
        titles1 = [r["title"] for r in data1["recommendations"]]
        titles2 = [r["title"] for r in data2["recommendations"]]
        assert titles1 == titles2
    
    def test_confidence_scores(self, auth_headers):
        """Test que les scores de confiance sont valides"""
        client.post("/model/train", json={}, headers=auth_headers)
        
        response = client.post("/recommend/ml",
                              json={"query": "Platformer Indie", "k": 10},
                              headers=auth_headers)
        data = response.json()
        
        # Vérifier que les scores sont décroissants
        confidences = [r["confidence"] for r in data["recommendations"]]
        assert all(0 <= c <= 1 for c in confidences)
        assert confidences == sorted(confidences, reverse=True)
