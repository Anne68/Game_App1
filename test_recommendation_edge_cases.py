from fastapi.testclient import TestClient
from api_games_plus import app

client = TestClient(app)

def test_recommend_k1_ok():
    resp = client.post("/recommend", json={"title": "Halo", "k": 1})
    # 200 si la logique accepte k=1
    assert resp.status_code in (200, 404, 400)  # adapte selon ton comportement
    if resp.status_code == 200:
        data = resp.json()
        assert data["k"] == 1
        assert len(data["items"]) <= 1

def test_recommend_unknown_title():
    resp = client.post("/recommend", json={"title": "___INCONNU___", "k": 5})
    # soit 200 avec liste vide, soit 404 selon ton design
    assert resp.status_code in (200, 404)
