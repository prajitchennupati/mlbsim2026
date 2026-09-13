from __future__ import annotations

import numpy as np
import pytest

from mlbsim_engine.game import simulate_game
from mlbsim_engine.outcomes import HR, LEAGUE_RATES, K


def _lineup(scale_idx: int, factor: float):
    row = LEAGUE_RATES.copy()
    row[scale_idx] *= factor
    row /= row.sum()
    return np.tile(row, (9, 1))


def test_deterministic_for_a_seed():
    a = simulate_game(n_sims=2000, seed=42)
    b = simulate_game(n_sims=2000, seed=42)
    np.testing.assert_array_equal(a.home_score, b.home_score)
    np.testing.assert_array_equal(a.away_score, b.away_score)


def test_neutral_game_is_realistic():
    r = simulate_game(n_sims=15000, seed=7)
    s = r.summary()
    assert 0.49 <= s["home_win_prob"] <= 0.55  # home bats last -> small edge
    assert 7.5 <= s["exp_total_runs"] <= 10.0
    assert 0.05 <= s["p_extra_innings"] <= 0.15
    assert 0.02 <= s["p_shutout_home"] <= 0.12
    assert 0.20 <= s["p_one_run_game"] <= 0.38


def test_inning_runs_sum_to_final_score():
    r = simulate_game(n_sims=3000, seed=3)
    np.testing.assert_array_equal(r.inn_home.sum(axis=1), r.home_score)
    np.testing.assert_array_equal(r.inn_away.sum(axis=1), r.away_score)


def test_better_offense_wins_more_and_scores_more():
    strong = _lineup(HR, 2.5)  # lots of power
    weak = _lineup(K, 1.7)  # strikeout-prone
    r = simulate_game(home_lineup=strong, away_lineup=weak, n_sims=8000, seed=11)
    assert r.home_win_prob > 0.72
    assert r.home_score.mean() > r.away_score.mean() + 1.5


def test_home_does_not_bat_in_ninth_when_already_ahead():
    # give the home team a monster lineup so it usually leads after 8.5
    r = simulate_game(home_lineup=_lineup(HR, 3.0), n_sims=5000, seed=5)
    # 9th-inning (col 8) home runs should be near zero relative to earlier innings
    ninth = r.inn_home[:, 8].mean()
    third = r.inn_home[:, 2].mean()
    assert ninth < 0.6 * third


def test_chunked_run_matches_single_cohort_shapes_and_stats():
    from mlbsim_engine.game import _CHUNK

    n = _CHUNK * 2 + 137  # forces 3 cohorts incl. a short tail
    r = simulate_game(n_sims=n, seed=3, track_batting=True)
    assert r.n_sims == n
    assert r.home_score.shape == (n,) and r.away_score.shape == (n,)
    assert r.inn_home.shape == (n, 12)
    assert r.bat_home["pa"].shape == (9, n)
    assert r.sp_home["bf"].shape == (n,)
    # per-inning runs still reconcile to the final score for every sim
    np.testing.assert_array_equal(r.inn_home.sum(axis=1), r.home_score)
    # statistics land in the same place as a single small cohort
    small = simulate_game(n_sims=8000, seed=3, track_batting=False)
    assert abs(r.home_win_prob - small.home_win_prob) < 0.03
    assert abs(r.home_score.mean() - small.home_score.mean()) < 0.3


def test_summary_has_expected_shape():
    s = simulate_game(n_sims=1000, seed=1).summary()
    for key in (
        "home_win_prob",
        "exp_home_runs",
        "p_extra_innings",
        "most_likely_scores",
        "inning_probs",
        "home_score_dist",
        "total_runs_dist",
        "home_batting",
        "home_batting_props",
        "home_pitcher_props",
    ):
        assert key in s
    ip = s["inning_probs"]
    assert len(ip["home"]) == 12 and len(ip["p_home_lead_after"]) == 12
    assert set(ip["home"][0]) == {
        "exp_runs",
        "p_score_1plus",
        "p_score_2plus",
        "p_score_3plus",
        "p_scoreless",
    }
    assert sum(float(v) for v in s["home_score_dist"].values()) == pytest.approx(1.0, abs=1e-3)
    slot0 = s["away_batting"][0]
    assert 3.5 < slot0["pa"] < 5.5  # leadoff hitter PA per game
    # batter props are probabilities
    p0 = s["away_batting_props"][0]
    assert 0.0 <= p0["p_1plus_h"] <= 1.0
    assert p0["p_1plus_h"] >= p0["p_2plus_h"] >= p0["p_3plus_h"]
    # starter faces most of the lineup and racks up strikeouts
    sp = s["home_pitcher_props"]
    assert 3.0 < sp["mean_ip"] < 6.5
    assert sp["p_5plus_k"] >= sp["p_10plus_k"]
