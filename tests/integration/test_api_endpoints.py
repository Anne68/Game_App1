from fastapi.testclient import TestClient
from api_games_plus import app

client = TestClient(app)

def test_health_and_ready():
    assert client.get("/games/healthz").status_code == 200
    assert client.get("/games/ready").status_code == 200

def test_predict_flow():
    r_tok = client.post("/token", data={"username":"u","password":"p"})
    token = r_tok.json()["access_token"]
    r = client.post("/games/predict",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"title":"CodeGame","platform":"pc"})
    assert r.status_code == 200
    js = r.json()
    assert "label" in js and "score" in js
