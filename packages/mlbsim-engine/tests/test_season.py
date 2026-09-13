from __future__ import annotations

import time

import numpy as np
import pytest

from mlbsim_engine.season import SeasonSetup, simulate_season


def _setup(seed: int = 0) -> SeasonSetup:
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 0.08, 30)
    league = np.array([0] * 15 + [1] * 15, dtype=np.int64)
    division = np.tile(np.repeat([0, 1, 2], 5), 2).astype(np.int64)
    # standings-to-date track strength (stronger teams start ahead) plus noise
    wins_to_date = np.clip(47 + np.round(strength * 90) + rng.integers(-4, 5, 30), 30, 62).astype(
        np.int64
    )
    rem_home, rem_away = [], []
    for _ in range(72 * 30 // 2):
        a, b = rng.choice(30, 2, replace=False)
        rem_home.append(a)
        rem_away.append(b)
    rem_home = np.array(rem_home, dtype=np.int64)
    rem_away = np.array(rem_away, dtype=np.int64)
    d = strength[rem_home] - strength[rem_away]
    rem_p_home = 1.0 / (1.0 + np.exp(-(d + 0.15)))
    return SeasonSetup(
        team_ids=list(range(100, 130)),
        league=league,
        division=division,
        wins_to_date=wins_to_date,
        losses_to_date=(90 - wins_to_date).astype(np.int64),
        strength=strength,
        rem_home=rem_home,
        rem_away=rem_away,
        rem_p_home=rem_p_home,
    )


def test_probability_totals_are_exact():
    res = simulate_season(_setup(), n_sims=20000, seed=1)
    rows = res.summary()
    assert sum(r["p_playoffs"] for r in rows) == pytest.approx(12.0, abs=0.02)
    assert sum(r["p_division"] for r in rows) == pytest.approx(6.0, abs=0.02)
    assert sum(r["p_bye"] for r in rows) == pytest.approx(4.0, abs=0.02)
    assert sum(r["p_pennant"] for r in rows) == pytest.approx(2.0, abs=0.02)
    assert sum(r["p_world_series"] for r in rows) == pytest.approx(1.0, abs=0.02)


def test_stronger_teams_reach_the_postseason_more():
    setup = _setup(2)
    res = simulate_season(setup, n_sims=20000, seed=2)
    rows = res.summary()
    by_id = {r["team_id"]: r for r in rows}
    order = np.argsort(setup.strength)
    weak = by_id[setup.team_ids[order[0]]]["p_playoffs"]
    strong = by_id[setup.team_ids[order[-1]]]["p_playoffs"]
    assert strong > weak
    # expected wins should roughly track strength ordering
    exp_wins = np.array([by_id[t]["exp_wins"] for t in setup.team_ids])
    assert np.corrcoef(exp_wins, setup.strength)[0, 1] > 0.6


def test_win_distribution_and_seed_expectations():
    res = simulate_season(_setup(3), n_sims=10000, seed=3)
    rows = res.summary()
    for r in rows:
        total = sum(r["win_dist"].values())
        assert total == 10000
        if r["exp_seed"] is not None:
            assert 1.0 <= r["exp_seed"] <= 6.0


def test_series_length_summary_is_sane():
    res = simulate_season(_setup(4), n_sims=15000, seed=4)
    s = res.series_length_summary()
    assert set(s) == {"WC", "DS", "LCS", "WS"}
    assert s["WS"]["best_of"] == 7
    assert 5.0 < s["WS"]["exp_games"] < 6.3
    assert 4.0 < s["DS"]["exp_games"] < 4.6
    assert abs(sum(s["WS"]["game_count_dist"].values()) - 1.0) < 0.01


@pytest.mark.bench
def test_100k_seasons_under_budget():
    setup = _setup(5)
    start = time.perf_counter()
    simulate_season(setup, n_sims=100_000, seed=5)
    elapsed = time.perf_counter() - start
    assert elapsed < 8.0, f"100k-season sim took {elapsed:.2f}s (budget 8s)"
