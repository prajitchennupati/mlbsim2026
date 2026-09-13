"""Plate-appearance outcome vocabulary and league baselines.

The engine models eight mutually exclusive PA outcomes. ``field_out`` is a ball
put in play that is fielded for an out; ``k`` is a strikeout.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

OUTCOMES: tuple[str, ...] = ("field_out", "k", "bb", "hbp", "1b", "2b", "3b", "hr")
N_OUTCOMES = len(OUTCOMES)

FIELD_OUT, K, BB, HBP, S1, S2, S3, HR = range(N_OUTCOMES)

# Approximate 2023-24 league PA-outcome rates (sum to 1). Used as the odds-ratio
# neutraliser and as the fallback for players with no data.
# Tuned so two neutral lineups produce ~9.1 combined runs / game (roughly the
# recent MLB average); revisited as the engine is calibrated against M2.
LEAGUE_RATES: NDArray[np.float64] = np.array(
    [
        0.448,  # field_out (BIP out; includes reached-on-error, fielder's choice)
        0.228,  # k
        0.082,  # bb
        0.011,  # hbp
        0.146,  # 1b
        0.045,  # 2b
        0.004,  # 3b
        0.030,  # hr
    ],
    dtype=np.float64,
)
LEAGUE_RATES = LEAGUE_RATES / LEAGUE_RATES.sum()

# Runners-advance tunables (rule-based baserunning; see baserunning.py).
P_SAC_SCORE_FROM_3B = 0.32  # productive out / sac fly, < 2 outs
P_SCORE_FROM_2B_ON_1B = 0.60
P_1B_RUNNER_TO_3B_ON_1B = 0.28
P_SCORE_FROM_1B_ON_2B = 0.45
