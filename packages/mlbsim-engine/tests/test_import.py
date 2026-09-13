import mlbsim_engine


def test_engine_exposes_public_api():
    assert mlbsim_engine.__version__ == "0.0.0"
    assert mlbsim_engine.ENGINE_VERSION == "0.1.0"
    for name in ("simulate_game", "GameSimResult", "advance_batch", "odds_ratio_matchup"):
        assert hasattr(mlbsim_engine, name)
