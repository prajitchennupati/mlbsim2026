"""Sabermetric constants for descriptive rate stats.

These are league-average coefficients that drift year to year. M1 uses a single
representative set (FanGraphs "Guts!", 2023) for all seasons; making them
season-specific is a documented M2 refinement. They affect only descriptive
columns in ``*_season_stats`` — no model consumes them directly.
"""

from __future__ import annotations

# wOBA linear weights (2023).
WOBA_WEIGHTS = {
    "bb": 0.696,
    "hbp": 0.726,
    "b1": 0.883,
    "b2": 1.244,
    "b3": 1.569,
    "hr": 2.004,
}
WOBA_SCALE = 1.204

# FIP constant (2023): FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + FIP_CONSTANT
FIP_CONSTANT = 3.10
