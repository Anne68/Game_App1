from api_games_plus import predict_one

def test_predict_output_schema():
    req = {"title": "The Legend of Code", "platform": "pc"}
    out = predict_one(req)
    assert set(out.keys()) == {"label", "score"}
    assert 0.0 <= out["score"] <= 1.0
