"""Turn raw stat lines into PA-outcome rate vectors and combine them via log5.

The odds-ratio ("log5") method (Tango): for each outcome, combine the batter's
rate, the pitcher's rate, and the league rate on the odds scale, then renormalise
to a valid distribution. It is a strong, low-variance baseline; a learned event
model (mlbsim-models) can replace the per-player rate vectors later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mlbsim_engine.outcomes import LEAGUE_RATES

Array = NDArray[np.float64]
_EPS = 1e-9


def rates_from_stat_json(stat: dict[str, Any], *, is_pitcher: bool = False) -> Array:
    """Best-effort PA-outcome rates from a ``player_season_stats`` batting/pitching blob.

    Falls back to the league vector for anything missing. Returns a length-8
    probability vector in :data:`mlbsim_engine.outcomes.OUTCOMES` order.
    """
    pa = float(stat.get("pa") or stat.get("bf") or 0.0)
    if pa < 1.0:
        return LEAGUE_RATES.copy()

    def r(key: str, league_idx: int) -> float:
        v = stat.get(key)
        return float(v) / pa if v is not None else float(LEAGUE_RATES[league_idx])

    k = r("so", 1)
    bb = r("bb", 2)
    hbp = r("hbp", 3)
    hr = r("hr", 7)
    d2 = r("d2b", 5) if not is_pitcher else float(LEAGUE_RATES[5])
    t3 = r("t3b", 6) if not is_pitcher else float(LEAGUE_RATES[6])
    h = (
        float(stat["h"]) / pa
        if stat.get("h") is not None
        else float(LEAGUE_RATES[4] + LEAGUE_RATES[5] + LEAGUE_RATES[6] + LEAGUE_RATES[7])
    )
    s1 = max(h - d2 - t3 - hr, 0.0)
    non_out = k + bb + hbp + s1 + d2 + t3 + hr
    field_out = max(1.0 - non_out, 0.02)
    vec = np.array([field_out, k, bb, hbp, s1, d2, t3, hr], dtype=np.float64)
    return vec / vec.sum()


def odds_ratio_matchup(batter: Array, pitcher: Array, league: Array | None = None) -> Array:
    """log5 combine per-outcome rates. Accepts (8,) or (n, 8) arrays (broadcast)."""
    lg = LEAGUE_RATES if league is None else league
    b = np.clip(np.asarray(batter, dtype=np.float64), _EPS, 1.0 - _EPS)
    p = np.clip(np.asarray(pitcher, dtype=np.float64), _EPS, 1.0 - _EPS)
    lgc = np.clip(np.asarray(lg, dtype=np.float64), _EPS, 1.0 - _EPS)

    def odds(x: Array) -> Array:
        return x / (1.0 - x)

    combined_odds = odds(b) * odds(p) / odds(lgc)
    prob = combined_odds / (1.0 + combined_odds)
    total = prob.sum(axis=-1, keepdims=True)
    return np.asarray(prob / total, dtype=np.float64)


def lineup_matrix(stat_jsons: list[dict[str, Any]]) -> Array:
    """Stack nine batter stat blobs into a (9, 8) rate matrix (pads/truncates)."""
    rows = [rates_from_stat_json(s) for s in stat_jsons[:9]]
    while len(rows) < 9:
        rows.append(LEAGUE_RATES.copy())
    return np.vstack(rows).astype(np.float64)


def neutral_lineup() -> Array:
    return np.tile(LEAGUE_RATES, (9, 1)).astype(np.float64)


def neutral_pitcher() -> Array:
    return LEAGUE_RATES.copy()
