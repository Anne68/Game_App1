from api_games_plus import predict_one
def test_predict_output_schema():
    out = predict_one({"title":"abc","platform":"pc"})
    assert set(out)=={"label","score"}
    assert 0.0 <= out["score"] <= 1.0
