from fastapi.testclient import TestClient
from api_games_plus import app

client = TestClient(app)

def test_recommend_contract_validation():
    # k trop grand -> 422
    resp = client.post("/recommend", json={"title": "Halo", "k": 999})
    assert resp.status_code == 422

    # pas de titre -> 422
    resp = client.post("/recommend", json={"k": 5})
    assert resp.status_code == 422

def test_refresh_flow_requires_valid_refresh():
    # refresh sans token -> 422 (pydantic)
    resp = client.post("/token/refresh", json={})
    assert resp.status_code == 422
