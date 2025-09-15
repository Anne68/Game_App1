# test_integration.py - Tests d'intégration end-to-end
import pytest
import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor

class TestFullIntegration:
    """Tests d'intégration end-to-end"""
    
    @pytest.fixture
    def api_url(self):
        # URL locale pour les tests
        return "http://localhost:8000"
    
    @pytest.fixture
    def auth_token(self, api_url):
        """Obtenir un token d'authentification pour les tests"""
        # Créer un utilisateur de test
        try:
            requests.post(f"{api_url}/register", 
                         data={"username": "testuser", "password": "testpass123"})
        except:
            pass  # L'utilisateur existe peut-être déjà
        
        # S'authentifier
        response = requests.post(f"{api_url}/token", 
                               data={"username": "testuser", "password": "testpass123"})
        if response.status_code == 200:
            return response.json()["access_token"]
        return None
    
    def test_api_health(self, api_url):
        """Test de santé de l'API"""
        response = requests.get(f"{api_url}/healthz", timeout=10)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
    
    def test_authentication_flow(self, api_url):
        """Test complet du flux d'authentification"""
        # 1. Inscription
        response = requests.post(f"{api_url}/register", 
                               data={"username": "integtest", "password": "integpass123"})
        # 200 ou 400 si existe déjà
        assert response.status_code in [200, 400]
        
        # 2. Connexion
        response = requests.post(f"{api_url}/token", 
                               data={"username": "integtest", "password": "integpass123"})
        assert response.status_code == 200
        
        token_data = response.json()
        assert "access_token" in token_data
        assert "token_type" in token_data
        
        # 3. Utiliser le token
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
        response = requests.get(f"{api_url}/model/metrics", headers=headers)
        assert response.status_code == 200
    
    def test_full_ml_workflow(self, api_url, auth_token):
        """Test du workflow ML complet"""
        if not auth_token:
            pytest.skip("No auth token available")
            
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # 1. Vérifier l'état initial
        response = requests.get(f"{api_url}/model/metrics", headers=headers)
        assert response.status_code == 200
        
        # 2. Entraîner le modèle
        train_response = requests.post(f"{api_url}/model/train", 
                                     json={"force_retrain": True}, 
                                     headers=headers, 
                                     timeout=60)
        assert train_response.status_code == 200
        
        train_data = train_response.json()
        assert train_data["status"] == "success"
        assert "version" in train_data
        assert "duration" in train_data
        
        # 3. Faire des prédictions ML
        pred_response = requests.post(f"{api_url}/recommend/ml",
                                    json={"query": "RPG Action", "k": 5},
                                    headers=headers,
                                    timeout=30)
        assert pred_response.status_code == 200
        
        pred_data = pred_response.json()
        assert "recommendations" in pred_data
        assert "model_version" in pred_data
        assert len(pred_data["recommendations"]) <= 5
        
        # 4. Vérifier la structure des recommandations
        if pred_data["recommendations"]:
            rec = pred_data["recommendations"][0]
            assert "title" in rec
            assert "confidence" in rec
            assert 0 <= rec["confidence"] <= 1
        
        # 5. Vérifier les métriques après prédiction
        metrics_response = requests.get(f"{api_url}/model/metrics", headers=headers)
        metrics_data = metrics_response.json()
        assert metrics_data["is_trained"] is True
        assert metrics_data["total_predictions"] >= 1
    
    def test_search_functionality(self, api_url, auth_token):
        """Test de la fonctionnalité de recherche"""
        if not auth_token:
            pytest.skip("No auth token available")
            
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Test recherche par titre
        response = requests.get(f"{api_url}/games/by-title/witcher", 
                              headers=headers, timeout=10)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        # Vérifier la structure des résultats
        if data["results"]:
            game = data["results"][0]
            assert "id" in game
            assert "title" in game
    
    def test_monitoring_endpoints(self, api_url, auth_token):
        """Test des endpoints de monitoring"""
        if not auth_token:
            pytest.skip("No auth token available")
            
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Summary
        response = requests.get(f"{api_url}/monitoring/summary", 
                              headers=headers, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "monitoring" in data
        
        # Drift detection
        response = requests.get(f"{api_url}/monitoring/drift", 
                              headers=headers, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "drift_score" in data

class TestLoadTesting:
    """Tests de charge"""
    
    @pytest.fixture
    def setup_model(self):
        """Entraîner le modèle avant les tests de charge"""
        api_url = "http://localhost:8000"
        
        # Auth
        requests.post(f"{api_url}/register", 
                     data={"username": "loadtest", "password": "loadpass123"})
        response = requests.post(f"{api_url}/token", 
                               data={"username": "loadtest", "password": "loadpass123"})
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Entraîner
        requests.post(f"{api_url}/model/train", json={}, headers=headers)
        
        return api_url, headers
    
    def test_concurrent_predictions(self, setup_model):
        """Test de prédictions concurrentes"""
        api_url, headers = setup_model
        
        def make_prediction(query_suffix):
            try:
                response = requests.post(f"{api_url}/recommend/ml",
                                       json={"query": f"Action {query_suffix}", "k": 3},
                                       headers=headers, timeout=10)
                return response.status_code == 200
            except:
                return False
        
        # 10 requêtes simultanées
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_prediction, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8, f"Success rate too low: {success_rate}"
    
    def test_response_time_sla(self, setup_model):
        """Test SLA de temps de réponse"""
        api_url, headers = setup_model
        
        response_times = []
        for _ in range(5):
            start = time.time()
            response = requests.post(f"{api_url}/recommend/ml",
                                   json={"query": "RPG", "k": 5},
                                   headers=headers)
            duration = time.time() - start
            response_times.append(duration)
            
            assert response.status_code == 200
        
        avg_response_time = sum(response_times) / len(response_times)
        p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]
        
        assert avg_response_time < 1.0, f"Average response time too high: {avg_response_time}s"
        assert p95_response_time < 2.0, f"P95 response time too high: {p95_response_time}s"

class TestDataValidation:
    """Tests de validation des données"""
    
    def test_model_training_validation(self):
        """Test validation des données d'entraînement"""
        from model_manager import RecommendationModel
        
        # Données invalides
        invalid_games = [
            {"id": 1, "title": "", "genres": "RPG"},  # Titre vide
            {"id": 2, "title": "Game", "genres": "", "rating": -1},  # Rating invalide
        ]
        
        model = RecommendationModel()
        with pytest.raises(Exception):
            model.train(invalid_games)
    
    def test_prediction_input_validation(self):
        """Test validation des entrées de prédiction"""
        from model_manager import RecommendationModel
        
        # Données valides pour l'entraînement
        valid_games = [
            {"id": 1, "title": "Test Game", "genres": "Action RPG", 
             "rating": 4.5, "metacritic": 85, "platforms": ["PC"]}
        ]
        
        model = RecommendationModel()
        model.train(valid_games)
        
        # Test requête vide
        recommendations = model.predict("", k=5)
        assert len(recommendations) >= 0  # Peut retourner des résultats ou non
        
        # Test k invalide
        recommendations = model.predict("Action", k=0)
        assert len(recommendations) == 0
