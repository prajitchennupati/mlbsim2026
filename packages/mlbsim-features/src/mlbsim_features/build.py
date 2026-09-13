"""Point-in-time game feature vectors.

The pure functions (``team_form``, ``pitcher_form``, ``assemble_game_features``)
take already-filtered, chronologically-sorted history and are fully
unit-testable. ``build_game_features`` wires them to the warehouse and persists
``warehouse.game_features`` rows.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from mlbsim_features.asof import _as_date, before

FEATURE_SET_ID = "fs_v1"

# Neutral fallbacks for teams / pitchers with little or no history yet.
_NEUTRAL_TEAM = {
    "win_pct": 0.5,
    "rs_pg": 4.5,
    "ra_pg": 4.5,
    "run_diff_pg": 0.0,
    "win_pct_l10": 0.5,
    "run_diff_pg_l10": 0.0,
    "rest_days": 3.0,
    "games_played": 0.0,
}
_NEUTRAL_SP = {
    "sp_era": 4.20,
    "sp_fip": 4.20,
    "sp_k_pct": 0.22,
    "sp_bb_pct": 0.08,
    "sp_ip_per_start": 5.2,
    "sp_era_l3": 4.20,
    "sp_starts": 0.0,
}


def _mean(xs: list[float], default: float) -> float:
    return sum(xs) / len(xs) if xs else default


def team_form(history: list[dict[str, Any]], as_of: dt.date | str) -> dict[str, float]:
    """Team form as of a date, from that team's prior game logs this season.

    Each history row needs ``game_date``, ``runs_for``, ``runs_against``, ``won``.
    """
    rows = before(history, as_of, date_key="game_date")
    if not rows:
        return dict(_NEUTRAL_TEAM)
    n = len(rows)
    rf = [float(r["runs_for"]) for r in rows]
    ra = [float(r["runs_against"]) for r in rows]
    wins = [1.0 if r["won"] else 0.0 for r in rows]
    last10 = rows[-10:]
    rf10 = [float(r["runs_for"]) for r in last10]
    ra10 = [float(r["runs_against"]) for r in last10]
    w10 = [1.0 if r["won"] else 0.0 for r in last10]
    rest = (_as_date(as_of) - _as_date(rows[-1]["game_date"])).days
    return {
        "win_pct": _mean(wins, 0.5),
        "rs_pg": _mean(rf, 4.5),
        "ra_pg": _mean(ra, 4.5),
        "run_diff_pg": _mean(rf, 4.5) - _mean(ra, 4.5),
        "win_pct_l10": _mean(w10, 0.5),
        "run_diff_pg_l10": _mean(rf10, 4.5) - _mean(ra10, 4.5),
        "rest_days": float(min(rest, 10)),
        "games_played": float(n),
    }


def pitcher_form(history: list[dict[str, Any]], as_of: dt.date | str) -> dict[str, float]:
    """Starter form as of a date, from that pitcher's prior game logs this season.

    Each history row needs ``game_date``, ``outs``, ``er``, ``so``, ``bb``,
    ``hbp``, ``hr``, ``bf``, ``is_start``.
    """
    rows = [r for r in before(history, as_of, date_key="game_date") if r.get("is_start")]
    if not rows:
        return dict(_NEUTRAL_SP)
    outs = sum(int(r["outs"]) for r in rows)
    ip = outs / 3 or 1e-9
    er = sum(int(r["er"]) for r in rows)
    so = sum(int(r["so"]) for r in rows)
    bb = sum(int(r["bb"]) for r in rows)
    hbp = sum(int(r["hbp"]) for r in rows)
    hr = sum(int(r["hr"]) for r in rows)
    bf = sum(int(r["bf"]) for r in rows) or 1
    last3 = rows[-3:]
    outs3 = sum(int(r["outs"]) for r in last3) or 1
    er3 = sum(int(r["er"]) for r in last3)
    return {
        "sp_era": 9.0 * er / ip,
        "sp_fip": (13 * hr + 3 * (bb + hbp) - 2 * so) / ip + 3.10,
        "sp_k_pct": so / bf,
        "sp_bb_pct": bb / bf,
        "sp_ip_per_start": ip / len(rows),
        "sp_era_l3": 9.0 * er3 / (outs3 / 3),
        "sp_starts": float(len(rows)),
    }


# Stats where the away side should be subtracted (higher = better for that team).
_HIGHER_BETTER = {
    "win_pct",
    "rs_pg",
    "run_diff_pg",
    "win_pct_l10",
    "run_diff_pg_l10",
    "rest_days",
    "sp_k_pct",
    "sp_ip_per_start",
}
# Stats where lower = better; diff is still home - away but the model learns the sign.
_LOWER_BETTER = {"ra_pg", "sp_era", "sp_fip", "sp_bb_pct", "sp_era_l3"}


def assemble_game_features(
    home_team: dict[str, float],
    away_team: dict[str, float],
    home_sp: dict[str, float],
    away_sp: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Return {'home': raw, 'away': raw, 'diff': home - away} feature dicts."""
    home_raw = {**home_team, **home_sp}
    away_raw = {**away_team, **away_sp}
    diff = {
        f"d_{k}": round(home_raw[k] - away_raw[k], 5)
        for k in (_HIGHER_BETTER | _LOWER_BETTER)
        if k in home_raw and k in away_raw
    }
    return {
        "home": {k: round(v, 5) for k, v in home_raw.items()},
        "away": {k: round(v, 5) for k, v in away_raw.items()},
        "diff": diff,
    }


DIFF_FEATURE_ORDER: tuple[str, ...] = tuple(
    f"d_{k}" for k in sorted(_HIGHER_BETTER | _LOWER_BETTER)
)


def feature_vector(diff: dict[str, float]) -> list[float]:
    """Diff dict → ordered float list matching :data:`DIFF_FEATURE_ORDER`."""
    return [float(diff.get(k, 0.0)) for k in DIFF_FEATURE_ORDER]
