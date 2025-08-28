from fastapi.testclient import TestClient
from api_games_plus import app
c = TestClient(app)
def test_healthz(): assert c.get("/games/healthz").status_code == 200
