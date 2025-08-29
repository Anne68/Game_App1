# test_performance.py - Tests de performance
import pytest
import time
import psutil
import os
import threading
from model_manager import RecommendationModel
from fastapi.testclient import TestClient
from api_games_plus import app

class TestModelPerformance:
    """Tests de performance du modèle"""
    
    def test_training_time(self):
        """Test que l'entraînement est rapide"""
        # Dataset de taille moyenne pour le test
        games = [
            {"id": i, "title": f"Game {i}", "genres": "Action RPG", 
             "rating": 4.0 + (i % 10) * 0.1, "metacritic": 80 + (i % 20), 
             "platforms": ["PC", "PS4"]}
            for i in range(100)
        ]
        
        model = RecommendationModel()
        
        start = time.time()
        result = model.train(games)
        duration = time.time() - start
        
        assert result["status"] == "success"
        assert duration < 10.0, f"Training took too long: {duration}s"
    
    def test_prediction_time(self):
        """Test que la prédiction est rapide"""
        # Petit dataset pour test rapide
        games = [
            {"id": i, "title": f"Game {i}", "genres": "Action RPG", 
             "rating": 4.5, "metacritic": 85, "platforms": ["PC"]}
            for i in range(10)
        ]
        
        model = RecommendationModel()
        model.train(games)
        
        # Test temps de prédiction
        prediction_times = []
        for _ in range(20):
            start = time.time()
            recommendations = model.predict("Action RPG", k=5)
            duration = time.time() - start
            prediction_times.append(duration)
        
        avg_time = sum(prediction_times) / len(prediction_times)
        max_time = max(prediction_times)
        
        assert avg_time < 0.1, f"Average prediction time too high: {avg_time}s"
        assert max_time < 0.5, f"Max prediction time too high: {max_time}s"
    
    def test_memory_usage(self):
        """Test d'utilisation mémoire"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Créer un dataset plus important pour tester la mémoire
        large_dataset = [
            {"id": i, "title": f"Game {i}", "genres": "Action RPG Adventure", 
             "rating": 4.0 + (i % 10) * 0.1, "metacritic": 80 + (i % 20), 
             "platforms": ["PC", "PS4", "Xbox"]}
            for i in range(500)
        ]
        
        model = RecommendationModel()
        model.train(large_dataset)
        
        # Faire plusieurs prédictions pour tester les fuites mémoire
        for i in range(50):
            model.predict(f"Action {i}", k=10)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
        
        assert memory_increase < 200, f"Memory increase too high: {memory_increase}MB"
    
    def test_concurrent_predictions(self):
        """Test prédictions concurrentes"""
        games = [
            {"id": i, "title": f"Game {i}", "genres": "Action RPG", 
             "rating": 4.5, "metacritic": 85, "platforms": ["PC"]}
            for i in range(20)
        ]
        
        model = RecommendationModel()
        model.train(games)
        
        results = []
        errors = []
        
        def make_prediction(thread_id):
            try:
                start = time.time()
                recommendations = model.predict(f"Action {thread_id}", k=5)
                duration = time.time() - start
                results.append(duration)
            except Exception as e:
                errors.append(str(e))
        
        # Lancer 10 threads simultanés
        threads = []
        for i in range(10):
            t = threading.Thread(target=make_prediction, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors during concurrent predictions: {errors}"
        assert len(results) == 10, f"Not all predictions completed: {len(results)}"
        assert max(results) < 2.0, f"Slowest prediction too slow: {max(results)}s"

class TestAPIPerformance:
    """Tests de performance de l'API"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        # Créer un utilisateur de test
        client.post("/register", data={"username": "perfuser", "password": "perfpass123"})
        response = client.post("/token", data={"username": "perfuser", "password": "perfpass123"})
        if response.status_code == 200:
            token = response.json()["access_token"]
            return {"Authorization": f"Bearer {token}"}
        return {}
    
    def test_api_response_times(self, client, auth_headers):
        """Test des temps de réponse de l'API"""
        if not auth_headers:
            pytest.skip("No auth available")
        
        # Entraîner le modèle d'abord
        response = client.post("/model/train", json={}, headers=auth_headers)
        assert response.status_code == 200
        
        # Tester différents endpoints
        endpoints_tests = [
            ("GET", "/healthz", {}),
            ("GET", "/model/metrics", auth_headers),
            ("POST", "/recommend/ml", auth_headers, {"query": "RPG", "k": 5}),
            ("GET", "/games/by-title/test", auth_headers),
        ]
        
        for test in endpoints_tests:
            method, endpoint = test[0], test[1]
            headers = test[2] if len(test) > 2 else {}
            json_data = test[3] if len(test) > 3 else None
            
            start = time.time()
            if method == "GET":
                response = client.get(endpoint, headers=headers)
            else:
                response = client.post(endpoint, headers=headers, json=json_data)
            duration = time.time() - start
            
            assert response.status_code in [200, 201], f"Failed: {method} {endpoint}"
            assert duration < 2.0, f"Too slow: {method} {endpoint} took {duration}s"
    
    def test_concurrent_api_requests(self, client, auth_headers):
        """Test requêtes API concurrentes"""
        if not auth_headers:
            pytest.skip("No auth available")
        
        # Entraîner le modèle
        client.post("/model/train", json={}, headers=auth_headers)
        
        import concurrent.futures
        
        def make_request():
            return client.post("/recommend/ml",
                             json={"query": "Action", "k": 3},
                             headers=auth_headers)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(15)]
            results = [f.result() for f in futures]
        
        success_count = sum(1 for r in results if r.status_code == 200)
        success_rate = success_count / len(results)
        
        assert success_rate >= 0.9, f"Success rate too low: {success_rate}"
    
    def test_large_k_value_performance(self, client, auth_headers):
        """Test performance avec grande valeur de k"""
        if not auth_headers:
            pytest.skip("No auth available")
        
        client.post("/model/train", json={}, headers=auth_headers)
        
        start = time.time()
        response = client.post("/recommend/ml",
                              json={"query": "RPG Action", "k": 50},
                              headers=auth_headers)
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 3.0, f"Large k request too slow: {duration}s"

class TestStressTest:
    """Tests de stress"""
    
    def test_sustained_load(self):
        """Test de charge soutenue"""
        client = TestClient(app)
        
        # Auth setup
        client.post("/register", data={"username": "stressuser", "password": "stresspass123"})
        response = client.post("/token", data={"username": "stressuser", "password": "stresspass123"})
        if response.status_code != 200:
            pytest.skip("No auth available")
        
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Entraîner le modèle
        client.post("/model/train", json={}, headers=headers)
        
        # Test de charge soutenue pendant 30 secondes
        start_time = time.time()
        request_count = 0
        errors = 0
        response_times = []
        
        while time.time() - start_time < 30:  # 30 secondes
            req_start = time.time()
            try:
                response = client.post("/recommend/ml",
                                     json={"query": f"Action {request_count}", "k": 5},
                                     headers=headers)
                if response.status_code != 200:
                    errors += 1
            except Exception:
                errors += 1
            
            response_times.append(time.time() - req_start)
            request_count += 1
            time.sleep(0.1)  # 10 req/sec
        
        error_rate = errors / request_count
        avg_response_time = sum(response_times) / len(response_times)
        
        assert error_rate < 0.05, f"Error rate too high: {error_rate}"
        assert avg_response_time < 1.0, f"Average response time too high: {avg_response_time}s"
        assert request_count > 200, f"Not enough requests processed: {request_count}"

class TestResourceMonitoring:
    """Tests de monitoring des ressources"""
    
    def test_cpu_usage(self):
        """Test d'utilisation CPU"""
        # Mesurer l'utilisation CPU pendant l'entraînement
        games = [
            {"id": i, "title": f"Game {i}", "genres": "Action RPG Strategy", 
             "rating": 4.0 + (i % 10) * 0.1, "metacritic": 80 + (i % 20), 
             "platforms": ["PC", "PS4", "Xbox", "Switch"]}
            for i in range(200)
        ]
        
        model = RecommendationModel()
        
        # Mesurer CPU avant
        cpu_before = psutil.cpu_percent(interval=1)
        
        # Entraîner
        start_time = time.time()
        model.train(games)
        training_duration = time.time() - start_time
        
        # Mesurer CPU après
        cpu_after = psutil.cpu_percent(interval=1)
        
        # Le CPU ne devrait pas être bloqué à 100%
        assert cpu_after < 95, f"CPU usage too high: {cpu_after}%"
        assert training_duration < 30, f"Training took too long: {training_duration}s"
    
    def test_memory_leak_detection(self):
        """Test de détection de fuites mémoire"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        games = [
            {"id": i, "title": f"Game {i}", "genres": "Action", 
             "rating": 4.5, "metacritic": 85, "platforms": ["PC"]}
            for i in range(50)
        ]
        
        # Faire plusieurs cycles d'entraînement/prédiction
        memory_measurements = []
        
        for cycle in range(5):
            model = RecommendationModel(model_version=f"test-{cycle}")
            model.train(games)
            
            # Faire plusieurs prédictions
            for i in range(20):
                model.predict(f"Action {i}", k=10)
            
            current_memory = process.memory_info().rss / (1024 * 1024)
            memory_measurements.append(current_memory)
            
            # Nettoyer explicitement
            del model
        
        # Vérifier qu'il n'y a pas de fuite mémoire excessive
        memory_growth = memory_measurements[-1] - memory_measurements[0]
        assert memory_growth < 100, f"Potential memory leak: {memory_growth}MB growth"
    
    def test_database_connection_pool(self):
        """Test du pool de connexions DB (si applicable)"""
        from api_games_plus import get_db_conn
        
        connections = []
        try:
            # Créer plusieurs connexions simultanées
            for i in range(10):
                conn = get_db_conn()
                connections.append(conn)
                
                # Test basique de la connexion
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    assert result is not None
            
        except Exception as e:
            pytest.fail(f"Database connection pool test failed: {e}")
        finally:
            # Nettoyer les connexions
            for conn in connections:
                try:
                    conn.close()
                except:
                    pass
