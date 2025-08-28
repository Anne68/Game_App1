from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def _get_token():
    r = client.post("/token", data={"username": "u", "password": "p"})
    assert r.status_code == 200
    return r.json()["access_token"]

def test_healthz_ready():
    assert client.get("/healthz").status_code == 200
    assert client.get("/ready").status_code == 200

def test_predict_flow():
    token = _get_token()
    r = client.post("/games/predict",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"title": "The Legend of Code", "platform": "pc"})
    assert r.status_code == 200
    js = r.json()
    assert "label" in js and "score" in js
