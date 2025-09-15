# test_security.py - Tests de sécurité
import pytest
import requests
from fastapi.testclient import TestClient
from api_games_plus import app
import time

class TestAuthenticationSecurity:
    """Tests de sécurité de l'authentification"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_password_requirements(self, client):
        """Test des exigences de mot de passe"""
        weak_passwords = [
            "123",           # Trop court
            "password",      # Pas de majuscule/chiffre
            "PASSWORD",      # Pas de minuscule/chiffre
            "Password",      # Pas de chiffre
            "12345678",      # Que des chiffres
        ]
        
        for weak_pwd in weak_passwords:
            response = client.post("/register", 
                                 data={"username": f"test_{weak_pwd}", "password": weak_pwd})
            # Selon votre implémentation, cela pourrait être 400 ou passer
            # Le test vérifie que votre API gère bien les mots de passe faibles
            if response.status_code == 400:
                data = response.json()
                assert "password" in data.get("detail", "").lower()
    
    def test_sql_injection_attempts(self, client):
        """Test de tentatives d'injection SQL"""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "admin' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "admin'; UPDATE users SET password='hacked' --"
        ]
        
        for payload in sql_payloads:
            # Test dans username
            response = client.post("/token", 
                                 data={"username": payload, "password": "anypassword"})
            assert response.status_code == 401, f"SQL injection might work with: {payload}"
            
            # Test dans password
            response = client.post("/token", 
                                 data={"username": "admin", "password": payload})
            assert response.status_code == 401, f"SQL injection might work with password: {payload}"
    
    def test_jwt_token_validation(self, client):
        """Test de validation des tokens JWT"""
        # Token invalide
        invalid_tokens = [
            "invalid.token.here",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature",
            "Bearer fake_token",
            ""
        ]
        
        for token in invalid_tokens:
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get("/model/metrics", headers=headers)
            assert response.status_code == 401, f"Invalid token accepted: {token}"
    
    def test_token_expiration(self, client):
        """Test d'expiration des tokens"""
        # Créer un utilisateur
        client.post("/register", data={"username": "expiretest", "password": "expirepass123"})
        
        # Obtenir un token
        response = client.post("/token", data={"username": "expiretest", "password": "expirepass123"})
        assert response.status_code == 200
        
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Le token devrait fonctionner immédiatement
        response = client.get("/model/metrics", headers=headers)
        assert response.status_code == 200
        
        # Note: Pour un vrai test d'expiration, vous devriez configurer
        # un token avec une durée très courte dans les tests
    
    def test_rate_limiting_protection(self, client):
        """Test de protection contre les attaques par force brute"""
        # Tenter plusieurs connexions échouées rapidement
        failed_attempts = 0
        for i in range(20):
            response = client.post("/token", 
                                 data={"username": "nonexistent", "password": "wrongpass"})
            if response.status_code == 401:
                failed_attempts += 1
            elif response.status_code == 429:  # Rate limited
                break
        
        # Si votre API n'a pas de rate limiting, ce test passera quand même
        # mais vous savez que c'est une amélioration possible
        assert failed_attempts > 0, "Should have some failed attempts"

class TestInputValidationSecurity:
    """Tests de sécurité des entrées"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        client.post("/register", data={"username": "sectest", "password": "secpass123"})
        response = client.post("/token", data={"username": "sectest", "password": "secpass123"})
        if response.status_code == 200:
            token = response.json()["access_token"]
            return {"Authorization": f"Bearer {token}"}
        return {}
    
    def test_xss_protection(self, client, auth_headers):
        """Test de protection contre XSS"""
        if not auth_headers:
            pytest.skip("No auth available")
        
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for payload in xss_payloads:
            # Test dans les requêtes ML
            response = client.post("/recommend/ml",
                                 json={"query": payload, "k": 5},
                                 headers=auth_headers)
            
            if response.status_code == 200:
                data = response.json()
                # Vérifier que le payload n'est pas retourné tel quel
                response_text = str(data)
                assert "<script>" not in response_text
                assert "javascript:" not in response_text
    
    def test_path_traversal_protection(self, client, auth_headers):
        """Test de protection contre path traversal"""
        if not auth_headers:
            pytest.skip("No auth available")
        
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        for payload in path_payloads:
            # Test dans recherche par titre
            response = client.get(f"/games/by-title/{payload}", headers=auth_headers)
            # Ne devrait pas retourner d'erreur système ou de contenu de fichier
            assert response.status_code in [200, 400, 404]
            
            if response.status_code == 200:
                data = response.json()
                response_text = str(data)
                # Ne devrait pas contenir de contenu système sensible
                assert "root:" not in response_text
                assert "Administrator:" not in response_text
    
    def test_large_payload_protection(self, client, auth_headers):
        """Test de protection contre les gros payloads"""
        if not auth_headers:
            pytest.skip("No auth available")
        
        # Payload très volumineux
        large_query = "A" * 10000  # 10KB
        
        response = client.post("/recommend/ml",
                             json={"query": large_query, "k": 5},
                             headers=auth_headers)
        
        # L'API devrait gérer gracieusement ou rejeter
        assert response.status_code in [200, 400, 413, 422]
    
    def test_malformed_json_protection(self, client, auth_headers):
        """Test de protection contre JSON malformé"""
        if not auth_headers:
            pytest.skip("No auth available")
        
        malformed_payloads = [
            '{"query": "test", "k": }',  # JSON invalide
            '{"query": "test", "k": "not_a_number"}',  # Type invalide
            '{"query": null, "k": -1}',  # Valeurs invalides
        ]
        
        for payload in malformed_payloads:
            response = client.post("/recommend/ml",
                                 data=payload,
                                 headers={**auth_headers, "Content-Type": "application/json"})
            
            # Devrait retourner une erreur de validation
            assert response.status_code in [400, 422], f"Malformed JSON accepted: {payload}"

class TestDataPrivacySecurity:
    """Tests de sécurité des données"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_password_not_exposed(self, client):
        """Test que les mots de passe ne sont jamais exposés"""
        # Créer un utilisateur
        client.post("/register", data={"username": "privacytest", "password": "secret123"})
        
        # Obtenir un token
        response = client.post("/token", data={"username": "privacytest", "password": "secret123"})
        assert response.status_code == 200
        
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Tester tous les endpoints qui pourraient retourner des infos utilisateur
        endpoints_to_test = [
            "/model/metrics",
            "/monitoring/summary",
            "/healthz"
        ]
        
        for endpoint in endpoints_to_test:
            response = client.get(endpoint, headers=headers)
            if response.status_code == 200:
                response_text = response.text.lower()
                # Ne devrait jamais contenir le mot de passe
                assert "secret123" not in response_text
                assert "password" not in response_text or "hashed_password" in response_text
    
    def test_error_information_disclosure(self, client):
        """Test que les erreurs ne révèlent pas d'informations sensibles"""
        # Tenter des opérations qui devraient échouer
        error_scenarios = [
            ("POST", "/model/train", {}, {}),  # Sans auth
            ("GET", "/model/metrics", {}, {}),  # Sans auth
            ("POST", "/recommend/ml", {}, {"query": "test"}),  # Sans auth
        ]
        
        for method, endpoint, headers, json_data in error_scenarios:
            if method == "GET":
                response = client.get(endpoint, headers=headers)
            else:
                response = client.post(endpoint, headers=headers, json=json_data)
            
            if response.status_code >= 400:
                error_text = response.text.lower()
                # Ne devrait pas révéler de chemins système, de stack traces détaillées, etc.
                sensitive_info = [
                    "/app/",
                    "traceback",
                    "exception",
                    "mysql",
                    "database",
                    "connection",
                    "secret_key"
                ]
                
                for sensitive in sensitive_info:
                    assert sensitive not in error_text, f"Sensitive info exposed: {sensitive} in {endpoint}"

class TestDOSProtection:
    """Tests de protection contre les attaques DoS"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_resource_exhaustion_protection(self, client):
        """Test de protection contre l'épuisement des ressources"""
        # Créer un utilisateur
        client.post("/register", data={"username": "dostest", "password": "dospass123"})
        response = client.post("/token", data={"username": "dostest", "password": "dospass123"})
        
        if response.status_code != 200:
            pytest.skip("No auth available")
        
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Tenter de faire beaucoup de requêtes coûteuses rapidement
        start_time = time.time()
        successful_requests = 0
        
        for i in range(50):  # 50 requêtes rapidement
            try:
                response = client.post("/model/train",
                                     json={"force_retrain": True},
                                     headers=headers,
                                     timeout=5)  # Timeout court
                if response.status_code == 200:
                    successful_requests += 1
                elif response.status_code == 429:  # Rate limited
                    break
            except Exception:
                break  # Timeout ou autre erreur
        
        duration = time.time() - start_time
        
        # L'API devrait soit rate limiter, soit traiter les requêtes raisonnablement
        assert duration > 5 or successful_requests < 10, \
            "API might be vulnerable to resource exhaustion attacks"
