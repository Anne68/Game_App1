# test_performance.py — Tests de performance robustes pour la CI
import os
import gc
import time
import threading
import statistics
import concurrent.futures as futures

import pytest
import psutil
from fastapi.testclient import TestClient

from model_manager import RecommendationModel
from api_games_plus import app

# Active les tests lourds uniquement si l'ENV est positionnée
RUN_STRESS = os.getenv("RUN_STRESS") == "1"


# ----------------------------
# Tests de perf côté Modèle
# ----------------------------
class TestModelPerformance:
    """Tests de performance du modèle"""

    @pytest.mark.timeout(30)
    def test_training_time(self):
        """L'entraînement doit être raisonnablement rapide"""
        games = [
            {
                "id": i,
                "title": f"Game {i}",
                "genres": "Action RPG",
                "rating": 4.0 + (i % 10) * 0.1,
                "metacritic": 80 + (i % 20),
                "platforms": ["PC", "PS4"],
            }
            for i in range(100)
        ]

        model = RecommendationModel()

        t0 = time.perf_counter()
        result = model.train(games)
        dt = time.perf_counter() - t0

        assert result["status"] == "success"
        assert dt < 10.0, f"Training took too long: {dt:.2f}s"

    @pytest.mark.timeout(30)
    def test_prediction_time(self):
        """La prédiction doit être rapide"""
        games = [
            {
                "id": i,
                "title": f"Game {i}",
                "genres": "Action RPG",
                "rating": 4.5,
                "metacritic": 85,
                "platforms": ["PC"],
            }
            for i in range(10)
        ]

        model = RecommendationModel()
        model.train(games)

        # Warm-up
        for _ in range(3):
            model.predict("Action RPG", k=5)

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = model.predict("Action RPG", k=5)
            times.append(time.perf_counter() - t0)

        avg_t = statistics.fmean(times)
        max_t = max(times)

        assert avg_t < 0.10, f"Average prediction time too high: {avg_t:.3f}s"
        assert max_t < 0.50, f"Max prediction time too high: {max_t:.3f}s"

    @pytest.mark.timeout(60)
    def test_memory_usage(self):
        """Vérifie l'absence de fuite mémoire notable pendant l'inférence"""
        proc = psutil.Process(os.getpid())
        gc.collect()
        initial_rss = proc.memory_info().rss

        large_dataset = [
            {
                "id": i,
                "title": f"Game {i}",
                "genres": "Action RPG Adventure",
                "rating": 4.0 + (i % 10) * 0.1,
                "metacritic": 80 + (i % 20),
                "platforms": ["PC", "PS4", "Xbox"],
            }
            for i in range(500)
        ]

        model = RecommendationModel()
        model.train(large_dataset)

        for i in range(50):
            _ = model.predict(f"Action {i}", k=10)

        gc.collect()
        final_rss = proc.memory_info().rss
        delta_mb = (final_rss - initial_rss) / (1024 * 1024)

        # Marge tolérante pour runners partagés
        assert delta_mb < 200, f"Memory increase too high: +{delta_mb:.1f}MB"

    @pytest.mark.timeout(45)
    def test_concurrent_predictions(self):
        """Prédictions concurrentes (modèle en mémoire)"""
        games = [
            {
                "id": i,
                "title": f"Game {i}",
                "genres": "Action RPG",
                "rating": 4.5,
                "metacritic": 85,
                "platforms": ["PC"],
            }
            for i in range(20)
        ]

        model = RecommendationModel()
        model.train(games)

        results = []
        errors = []

        def make_prediction(thread_id: int):
            try:
                t0 = time.perf_counter()
                _ = model.predict(f"Action {thread_id}", k=5)
                results.append(time.perf_counter() - t0)
            except Exception as e:  # pragma: no cover
                errors.append(str(e))

        threads = [threading.Thread(target=make_prediction, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent predictions: {errors}"
        assert len(results) == 10, f"Not all predictions completed: {len(results)}"
        assert max(results) < 2.0, f"Slowest prediction too slow: {max(results):.2f}s"


# ----------------------------
# Tests de perf côté API
# ----------------------------
class TestAPIPerformance:
    """Tests de performance de l'API"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self, client):
        # Crée l'utilisateur si besoin, puis récupère un token
        client.post("/register", data={"username": "perfuser", "password": "perfpass123"})
        resp = client.post("/token", data={"username": "perfuser", "password": "perfpass123"})
        if resp.status_code == 200:
            token = resp.json()["access_token"]
            return {"Authorization": f"Bearer {token}"}
        return {}

    @pytest.mark.timeout(60)
    def test_api_response_times(self, client, auth_headers):
        """Temps de réponse des endpoints principaux"""
        if not auth_headers:
            pytest.skip("No auth available")

        # Entraîne le modèle d'abord
        r = client.post("/model/train", json={}, headers=auth_headers)
        assert r.status_code == 200

        endpoints = [
            ("GET", "/healthz", {}),
            ("GET", "/model/metrics", auth_headers),
            ("POST", "/recommend/ml", auth_headers, {"query": "RPG", "k": 5}),
            ("GET", "/games/by-title/test", auth_headers),
        ]

        for t in endpoints:
            method, endpoint = t[0], t[1]
            headers = t[2] if len(t) > 2 else {}
            payload = t[3] if len(t) > 3 else None

            t0 = time.perf_counter()
            resp = client.get(endpoint, headers=headers) if method == "GET" else client.post(
                endpoint, headers=headers, json=payload
            )
            dt = time.perf_counter() - t0

            assert resp.status_code in (200, 201, 204), f"Failed: {method} {endpoint}"
            assert dt < 2.0, f"Too slow: {method} {endpoint} took {dt:.3f}s"

    @pytest.mark.timeout(60)
    def test_concurrent_api_requests(self, client, auth_headers):
        """Requêtes API concurrentes — un client **par thread** (thread-safe)"""
        if not auth_headers:
            pytest.skip("No auth available")

        client.post("/model/train", json={}, headers=auth_headers)

        def make_request():
            # IMPORTANT : un client par thread
            with TestClient(app) as c:
                return c.post("/recommend/ml", json={"query": "Action", "k": 3}, headers=auth_headers)

        with futures.ThreadPoolExecutor(max_workers=5) as ex:
            results = list(ex.map(lambda _: make_request(), range(15)))

        success = sum(r.status_code in (200, 204) for r in results)
        rate = success / len(results)

        assert rate >= 0.90, f"Success rate too low: {rate:.2%}"

    @pytest.mark.timeout(60)
    def test_large_k_value_performance(self, client, auth_headers):
        """Performance avec grande valeur de k"""
        if not auth_headers:
            pytest.skip("No auth available")

        client.post("/model/train", json={}, headers=auth_headers)

        t0 = time.perf_counter()
        resp = client.post("/recommend/ml", json={"query": "RPG Action", "k": 50}, headers=auth_headers)
        dt = time.perf_counter() - t0

        assert resp.status_code in (200, 204)
        assert dt < 3.0, f"Large k request too slow: {dt:.3f}s"


# ----------------------------
# Stress test (désactivé par défaut)
# ----------------------------
class TestStressTest:
    """Tests de stress"""

    @pytest.mark.perf
    @pytest.mark.skipif(not RUN_STRESS, reason="Stress test disabled (set RUN_STRESS=1 to enable)")
    @pytest.mark.timeout(120)
    def test_sustained_load(self):
        """Charge soutenue (10 s)"""
        with TestClient(app) as client:
            client.post("/register", data={"username": "stressuser", "password": "stresspass123"})
            resp = client.post("/token", data={"username": "stressuser", "password": "stresspass123"})
            if resp.status_code != 200:
                pytest.skip("No auth available")

            token = resp.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            client.post("/model/train", json={}, headers=headers)

            start = time.time()
            errors = 0
            response_times = []
            req_count = 0

            # 10 s au lieu de 30 s pour rester raisonnable en CI
            while time.time() - start < 10.0:
                t0 = time.perf_counter()
                try:
                    r = client.post("/recommend/ml", json={"query": f"Action {req_count}", "k": 5}, headers=headers)
                    if r.status_code not in (200, 204):
                        errors += 1
                except Exception:
                    errors += 1
                response_times.append(time.perf_counter() - t0)
                req_count += 1
                time.sleep(0.1)  # ~10 req/s

            error_rate = errors / max(1, req_count)
            avg_rt = statistics.fmean(response_times)

            assert error_rate < 0.05, f"Error rate too high: {error_rate:.2%}"
            assert avg_rt < 1.0, f"Average response time too high: {avg_rt:.3f}s"
            assert req_count >= 80, f"Not enough requests processed: {req_count}"


# ----------------------------
# Monitoring des ressources
# ----------------------------
class TestResourceMonitoring:
    """Tests de monitoring des ressources"""

    @pytest.mark.timeout(60)
    def test_cpu_usage(self):
        """La charge CPU ne doit pas saturer durablement"""
        games = [
            {
                "id": i,
                "title": f"Game {i}",
                "genres": "Action RPG Strategy",
                "rating": 4.0 + (i % 10) * 0.1,
                "metacritic": 80 + (i % 20),
                "platforms": ["PC", "PS4", "Xbox", "Switch"],
            }
            for i in range(200)
        ]

        model = RecommendationModel()

        psutil.cpu_percent(interval=None)  # reset
        t0 = time.perf_counter()
        model.train(games)
        train_dt = time.perf_counter() - t0

        cpu_after = psutil.cpu_percent(interval=1.5)  # fenêtre un peu plus longue => mesure plus stable

        assert cpu_after < 98.0, f"CPU usage too high: {cpu_after:.1f}%"
        assert train_dt < 30.0, f"Training took too long: {train_dt:.2f}s"

    @pytest.mark.timeout(90)
    def test_memory_leak_detection(self):
        """Détection de fuite mémoire sur cycles train/predict"""
        proc = psutil.Process(os.getpid())
        gc.collect()
        base_mb = proc.memory_info().rss / (1024 * 1024)

        games = [
            {"id": i, "title": f"Game {i}", "genres": "Action", "rating": 4.5, "metacritic": 85, "platforms": ["PC"]}
            for i in range(50)
        ]

        measures = []
        for cycle in range(5):
            model = RecommendationModel(model_version=f"test-{cycle}")
            model.train(games)
            for i in range(20):
                _ = model.predict(f"Action {i}", k=10)

            gc.collect()
            measures.append(proc.memory_info().rss / (1024 * 1024))
            del model
            gc.collect()

        growth = measures[-1] - measures[0]
        assert growth < 100, f"Potential memory leak: {growth:.1f}MB growth"

    @pytest.mark.timeout(60)
    def test_database_connection_pool(self):
        """Test basique du pool de connexions DB (si applicable)"""
        from api_games_plus import get_db_conn

        conns = []
        try:
            for _ in range(10):
                conn = get_db_conn()
                conns.append(conn)
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    assert result is not None
        except Exception as e:
            pytest.fail(f"Database connection pool test failed: {e}")
        finally:
            for c in conns:
                try:
                    c.close()
                except Exception:
                    pass
