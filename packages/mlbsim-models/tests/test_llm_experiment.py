from __future__ import annotations

import numpy as np
import pytest

from mlbsim_models.llm.experiment import _parse_prob, compare_and_verdict, game_card


def test_game_card_is_deterministic_and_skips_blanks():
    ctx = {"home": "LAD", "away": "SFG", "date": "2026-04-01", "home_rs_pg": 5.1, "park": ""}
    card = game_card(ctx)
    assert game_card(ctx) == card
    assert "matchup: SFG @ LAD" in card
    assert "home_rs_pg: 5.1" in card
    assert "park" not in card  # blank value dropped


def test_parse_prob_from_json_and_from_loose_text():
    assert _parse_prob('{"home_win_prob": 0.63}') == pytest.approx(0.63)
    assert _parse_prob("I think about 0.58 chance") == pytest.approx(0.58)
    assert _parse_prob('{"home_win_prob": 1.4}') == pytest.approx(1.0 - 1e-4)  # clamped
    with pytest.raises(ValueError, match="no probability"):
        _parse_prob("no numbers here")


def test_compare_and_verdict_reports_loser():
    rng = np.random.default_rng(0)
    n = 500
    truth = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) < truth).astype(int)
    baseline = np.clip(truth + rng.normal(0, 0.03, n), 0.01, 0.99)  # sharp
    llm = np.full(n, 0.5)  # uninformative
    out = compare_and_verdict(llm, baseline, y)
    assert out["beats_baseline"] is False
    assert "does NOT beat" in out["verdict"]
    assert out["llm"]["log_loss"] > out["baseline"]["log_loss"]


def test_compare_and_verdict_reports_winner_when_llm_is_better():
    rng = np.random.default_rng(1)
    n = 500
    truth = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) < truth).astype(int)
    llm = np.clip(truth + rng.normal(0, 0.03, n), 0.01, 0.99)
    baseline = np.full(n, 0.5)
    out = compare_and_verdict(llm, baseline, y)
    assert out["beats_baseline"] is True
    assert "BEATS" in out["verdict"]
