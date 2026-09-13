"""Pure-function tests for the season-stat rate calculations (no database)."""

from __future__ import annotations

import pytest

from mlbsim_data.transform.season_stats import _batting_rates


def test_batting_rates_known_line():
    # 150-for-500, 30 2B / 3 3B / 25 HR, 60 BB, 5 HBP, 5 SF, 120 K
    singles = 150 - 30 - 3 - 25
    tb = singles + 2 * 30 + 3 * 3 + 4 * 25
    counts = {
        "pa": 570,
        "ab": 500,
        "h": 150,
        "d2b": 30,
        "t3b": 3,
        "hr": 25,
        "bb": 60,
        "ibb": 0,
        "hbp": 5,
        "so": 120,
        "sf": 5,
        "tb": tb,
    }
    r = _batting_rates(counts)
    assert r["avg"] == pytest.approx(0.300, abs=1e-3)
    assert r["obp"] == pytest.approx((150 + 60 + 5) / (500 + 60 + 5 + 5), abs=1e-3)
    assert r["slg"] == pytest.approx(tb / 500, abs=1e-3)
    assert r["ops"] == pytest.approx(r["obp"] + r["slg"], abs=1e-3)
    assert r["iso"] == pytest.approx(r["slg"] - 0.300, abs=1e-3)
    assert r["k_pct"] == pytest.approx(120 / 570, abs=1e-3)
    assert 0.30 < r["woba"] < 0.45  # plausible for a .300/.370/.500-ish hitter


def test_batting_rates_zero_ab_is_safe():
    counts = dict.fromkeys(
        ("pa", "ab", "h", "d2b", "t3b", "hr", "bb", "ibb", "hbp", "so", "sf", "tb"), 0
    )
    r = _batting_rates(counts)
    assert r["avg"] is None
    assert r["obp"] is None
    assert r["ops"] is None
