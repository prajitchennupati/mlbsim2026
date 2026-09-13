from __future__ import annotations

import numpy as np

from mlbsim_engine.game import GameState, live_win_probability, simulate_game


def test_start_state_defaults_match_a_fresh_game():
    a = simulate_game(n_sims=2000, seed=1)
    b = simulate_game(n_sims=2000, seed=1, start_state=GameState())
    np.testing.assert_array_equal(a.home_score, b.home_score)
    np.testing.assert_array_equal(a.away_score, b.away_score)


def test_resumed_game_keeps_the_current_score_as_a_floor():
    st = GameState(inning=5, half="bottom", outs=1, home_score=3, away_score=1)
    r = simulate_game(n_sims=3000, seed=2, start_state=st, track_batting=False)
    assert (r.home_score >= 3).all()
    assert (r.away_score >= 1).all()


def test_fresh_live_wp_is_near_even():
    p = live_win_probability(GameState(), n_sims=6000, seed=3)
    assert 0.47 <= p <= 0.56


def test_big_late_lead_is_almost_certain():
    st = GameState(inning=9, half="top", outs=0, home_score=8, away_score=2)
    p = live_win_probability(st, n_sims=4000, seed=4)
    assert p > 0.99


def test_trailing_late_is_almost_hopeless():
    st = GameState(inning=9, half="top", outs=2, home_score=1, away_score=6)
    p = live_win_probability(st, n_sims=4000, seed=5)
    assert p < 0.03


def test_walkoff_situation_favours_home():
    # bottom 9, tied, bases loaded, one out -> home is a heavy favourite
    st = GameState(inning=9, half="bottom", outs=1, base_state=0b111, home_score=4, away_score=4)
    p = live_win_probability(st, n_sims=6000, seed=6)
    assert p > 0.7
