from api_games_plus import create_access_token
def test_create_access_token():
    token = create_access_token("tester")
    assert isinstance(token, str) and len(token) > 10
