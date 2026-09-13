from __future__ import annotations

import numpy as np

from mlbsim_engine.playoffs import (
    HOME_BO3,
    HOME_BO7,
    game_prob_a,
    seed_league,
    simulate_series,
)


def test_game_prob_home_edge_and_monotone():
    s = np.zeros(1)
    assert 0.5 < game_prob_a(s, s, home_a=True)[0] < 0.56
    assert 0.44 < game_prob_a(s, s, home_a=False)[0] < 0.5
    gaps = [game_prob_a(np.array([d]), np.array([0.0]), home_a=True)[0] for d in (-0.3, 0, 0.3)]
    assert gaps == sorted(gaps)


def test_series_stronger_team_wins_more_and_length_bounds():
    rng = np.random.default_rng(0)
    n = 20000
    a = np.zeros(n, dtype=np.int64)
    b = np.ones(n, dtype=np.int64)
    strength = np.array([0.15, -0.15])  # A clearly better
    a_wins, games = simulate_series(a, b, strength, best_of=7, home_pattern=HOME_BO7, rng=rng)
    assert a_wins.mean() > 0.72
    assert games.min() >= 4 and games.max() <= 7
    assert 5.0 < games.mean() < 6.3  # realistic 7-game-series length


def test_bo3_sweep_rate_for_even_teams():
    rng = np.random.default_rng(1)
    n = 30000
    a = np.zeros(n, dtype=np.int64)
    b = np.ones(n, dtype=np.int64)
    _, games = simulate_series(a, b, np.zeros(2), best_of=3, home_pattern=HOME_BO3, rng=rng)
    assert games.min() == 2 and games.max() == 3
    assert 0.45 < (games == 2).mean() < 0.55  # ~half of even bo3 end in a sweep


def test_seed_league_produces_six_seeds_one_per_division_plus_wildcards():
    rng = np.random.default_rng(2)
    n = 5000
    t = 15
    division = np.repeat([0, 1, 2], 5)
    strength = rng.normal(0, 0.05, t)
    wins = rng.integers(70, 100, (n, t))
    res = seed_league(wins, strength, division, rng)

    seed_no = res["seed_no"]
    assert seed_no.shape == (n, t)
    # exactly six teams seeded per sim, seeds 1..6 each used once
    for row in seed_no[:50]:
        assert sorted(v for v in row if v > 0) == [1, 2, 3, 4, 5, 6]
    # each division contributes exactly one of seeds 1-3
    dw = res["division_winner"]
    for d in (0, 1, 2):
        assert np.all(dw[:, division == d].sum(axis=1) == 1)
    assert np.all(res["made"].sum(axis=1) == 6)


def test_seed_league_favours_strong_teams():
    rng = np.random.default_rng(3)
    n = 4000
    division = np.repeat([0, 1, 2], 5)
    strength = np.zeros(15)
    strength[0] = 0.30  # a juggernaut in division 0
    wins = rng.integers(78, 88, (n, 15))  # tight race so strength tiebreak matters
    res = seed_league(wins, strength, division, rng)
    assert res["made"][:, 0].mean() > res["made"][:, 1:5].mean()
