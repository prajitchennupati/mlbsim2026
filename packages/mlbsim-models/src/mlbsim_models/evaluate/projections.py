"""Per-stat error of a rate projection against a realised season.

Used for the Marcel backtest (project year N from N-3..N-1, score against actual
year N) and, later, the same comparison for Steamer / ZiPS.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mlbsim_engine.rates import rates_from_stat_json

_STATS = ("field_out", "k", "bb", "hbp", "1b", "2b", "3b", "hr")


def _rate_vec(stat: dict[str, Any], *, is_pitcher: bool) -> np.ndarray:
    return rates_from_stat_json(stat, is_pitcher=is_pitcher)


def projection_error(
    projected_rates: Sequence[np.ndarray],
    actual_lines: Sequence[dict[str, Any]],
    *,
    is_pitcher: bool = False,
) -> dict[str, Any]:
    """MAE / RMSE per outcome rate across a population of players.

    ``projected_rates[i]`` is a length-8 rate vector; ``actual_lines[i]`` is that
    player's realised season blob.
    """
    if len(projected_rates) != len(actual_lines):
        raise ValueError("projected_rates and actual_lines must be the same length")
    if not actual_lines:
        raise ValueError("no players to score")

    proj = np.vstack([np.asarray(p, dtype=np.float64) for p in projected_rates])
    act = np.vstack([_rate_vec(a, is_pitcher=is_pitcher) for a in actual_lines])
    err = proj - act

    per_stat = {
        name: {
            "mae": round(float(np.mean(np.abs(err[:, i]))), 5),
            "rmse": round(float(np.sqrt(np.mean(err[:, i] ** 2))), 5),
            "bias": round(float(np.mean(err[:, i])), 5),
        }
        for i, name in enumerate(_STATS)
    }
    return {
        "n_players": len(actual_lines),
        "overall_mae": round(float(np.mean(np.abs(err))), 5),
        "overall_rmse": round(float(np.sqrt(np.mean(err**2))), 5),
        "per_stat": per_stat,
    }
