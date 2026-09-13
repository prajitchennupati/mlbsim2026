from __future__ import annotations

import numpy as np
import pytest

from mlbsim_engine.outcomes import HR, LEAGUE_RATES, N_OUTCOMES, K
from mlbsim_engine.rates import (
    lineup_matrix,
    odds_ratio_matchup,
    rates_from_stat_json,
)


def test_neutral_matchup_returns_league():
    out = odds_ratio_matchup(LEAGUE_RATES, LEAGUE_RATES)
    np.testing.assert_allclose(out, LEAGUE_RATES, atol=1e-9)
    assert out.sum() == pytest.approx(1.0)


def test_strong_hitter_weak_pitcher_lifts_hr_rate():
    slugger = LEAGUE_RATES.copy()
    slugger[HR] *= 3
    slugger /= slugger.sum()
    meatball = LEAGUE_RATES.copy()
    meatball[HR] *= 2
    meatball /= meatball.sum()
    out = odds_ratio_matchup(slugger, meatball)
    assert out[HR] > slugger[HR] > LEAGUE_RATES[HR]
    assert out.sum() == pytest.approx(1.0)


def test_matchup_is_symmetric_in_batter_and_pitcher():
    a = LEAGUE_RATES.copy()
    a[K] *= 1.5
    a /= a.sum()
    b = LEAGUE_RATES.copy()
    b[K] *= 0.7
    b /= b.sum()
    np.testing.assert_allclose(odds_ratio_matchup(a, b), odds_ratio_matchup(b, a), atol=1e-12)


def test_vectorised_matchup_matches_rowwise():
    rng = np.random.default_rng(0)
    batters = rng.dirichlet(np.ones(N_OUTCOMES) * 5, size=20)
    pitcher = LEAGUE_RATES
    batch = odds_ratio_matchup(batters, np.tile(pitcher, (20, 1)))
    for i in range(20):
        np.testing.assert_allclose(batch[i], odds_ratio_matchup(batters[i], pitcher), atol=1e-9)


def test_rates_from_stat_json_shapes_and_fallback():
    empty = rates_from_stat_json({})
    np.testing.assert_allclose(empty, LEAGUE_RATES)
    line = rates_from_stat_json(
        {"pa": 600, "h": 165, "d2b": 32, "t3b": 2, "hr": 30, "bb": 60, "hbp": 6, "so": 140}
    )
    assert line.shape == (N_OUTCOMES,)
    assert line.sum() == pytest.approx(1.0)
    assert line[HR] == pytest.approx(30 / 600, abs=1e-6)


def test_lineup_matrix_pads_to_nine():
    m = lineup_matrix([{"pa": 500, "h": 150, "hr": 20, "so": 90, "bb": 45}] * 3)
    assert m.shape == (9, N_OUTCOMES)
    np.testing.assert_allclose(m.sum(axis=1), np.ones(9), atol=1e-9)
