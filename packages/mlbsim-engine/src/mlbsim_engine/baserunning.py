"""Vectorised rule-based baserunning.

Given the base-out state and the PA outcome for a cohort of simulated games,
compute the new base state, runs that scored, and outs recorded. The advancement
probabilities live in ``outcomes.py``; an empirical transition matrix learned
from play-by-play is a documented later refinement.

Base state is a 3-bit mask: bit 0 = runner on 1B, bit 1 = 2B, bit 2 = 3B.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mlbsim_engine.outcomes import (
    BB,
    FIELD_OUT,
    HBP,
    HR,
    P_1B_RUNNER_TO_3B_ON_1B,
    P_SAC_SCORE_FROM_3B,
    P_SCORE_FROM_1B_ON_2B,
    P_SCORE_FROM_2B_ON_1B,
    S1,
    S2,
    S3,
    K,
)

IntArr = NDArray[np.int64]
FloatArr = NDArray[np.float64]


def advance_batch(  # noqa: PLR0915
    base: IntArr,
    outs: IntArr,
    event: IntArr,
    u: FloatArr,
) -> tuple[IntArr, IntArr, IntArr]:
    """Return ``(new_base, runs_scored, outs_recorded)`` for each sim.

    ``u`` is an ``(n, 3)`` array of independent uniforms for the probabilistic
    advances. All inputs are 1-D of length ``n`` except ``u``.
    """
    n = base.shape[0]
    on1 = (base & 1) > 0
    on2 = (base & 2) > 0
    on3 = (base & 4) > 0

    runs = np.zeros(n, dtype=np.int64)
    outs_rec = np.zeros(n, dtype=np.int64)
    nb1 = np.zeros(n, dtype=bool)
    nb2 = np.zeros(n, dtype=bool)
    nb3 = np.zeros(n, dtype=bool)

    is_k = event == K
    is_out = event == FIELD_OUT
    is_bb = (event == BB) | (event == HBP)
    is_1b = event == S1
    is_2b = event == S2
    is_3b = event == S3
    is_hr = event == HR

    outs_rec[is_k | is_out] = 1

    # --- strikeout: nobody moves ---
    m = is_k
    nb1[m], nb2[m], nb3[m] = on1[m], on2[m], on3[m]

    # --- ball-in-play out: optional run from 3B (sac fly / productive out), <2 outs ---
    m = is_out
    sac = m & on3 & (outs < 2) & (u[:, 0] < P_SAC_SCORE_FROM_3B)
    runs[sac] += 1
    nb1[m], nb2[m] = on1[m], on2[m]
    nb3[m] = (on3 & ~sac)[m]

    # --- walk / HBP: forced advances only ---
    m = is_bb
    loaded = on1 & on2 & on3
    runs[m & loaded] += 1
    nb1[m] = True
    nb2[m] = (on1 | (on2 & ~on1))[m]
    nb3[m] = ((on1 & on2) | (on3 & ~(on1 & on2)))[m]

    # --- single ---
    m = is_1b
    runs[m & on3] += 1
    r2_score = m & on2 & (u[:, 1] < P_SCORE_FROM_2B_ON_1B)
    runs[r2_score] += 1
    r2_to3 = m & on2 & ~r2_score
    r1_to3 = m & on1 & (u[:, 2] < P_1B_RUNNER_TO_3B_ON_1B) & ~r2_to3
    r1_to2 = m & on1 & ~r1_to3
    nb1[m] = True
    nb2[m] = r1_to2[m]
    nb3[m] = (r2_to3 | r1_to3)[m]

    # --- double ---
    m = is_2b
    runs[m & on3] += 1
    runs[m & on2] += 1
    r1_score = m & on1 & (u[:, 1] < P_SCORE_FROM_1B_ON_2B)
    runs[r1_score] += 1
    r1_to3 = m & on1 & ~r1_score
    nb1[m] = False
    nb2[m] = True
    nb3[m] = r1_to3[m]

    # --- triple: everyone home, batter to 3B ---
    m = is_3b
    runs[m] += (on1.astype(np.int64) + on2 + on3)[m]
    nb1[m], nb2[m], nb3[m] = False, False, True

    # --- home run ---
    m = is_hr
    runs[m] += (1 + on1.astype(np.int64) + on2 + on3)[m]
    nb1[m], nb2[m], nb3[m] = False, False, False

    new_base = nb1.astype(np.int64) | (nb2.astype(np.int64) << 1) | (nb3.astype(np.int64) << 2)
    return new_base, runs, outs_rec
