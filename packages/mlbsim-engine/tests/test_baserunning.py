from __future__ import annotations

import numpy as np

from mlbsim_engine.baserunning import advance_batch
from mlbsim_engine.outcomes import BB, FIELD_OUT, HBP, HR, S1, S2, S3, K


def _run(base, outs, event, u=None):
    b = np.array([base], dtype=np.int64)
    o = np.array([outs], dtype=np.int64)
    e = np.array([event], dtype=np.int64)
    uu = np.array([u if u is not None else [0.99, 0.99, 0.99]], dtype=np.float64)
    nb, r, oa = advance_batch(b, o, e, uu)
    return int(nb[0]), int(r[0]), int(oa[0])


def test_strikeout_records_out_no_movement():
    assert _run(0b111, 1, K) == (0b111, 0, 1)


def test_home_run_clears_bases_and_scores_everyone():
    assert _run(0b111, 0, HR) == (0, 4, 0)
    assert _run(0, 2, HR) == (0, 1, 0)


def test_walk_forces_only():
    assert _run(0b000, 0, BB) == (0b001, 0, 0)  # batter to 1B
    assert _run(0b001, 0, BB) == (0b011, 0, 0)  # runner 1B->2B forced
    assert _run(0b011, 0, HBP) == (0b111, 0, 0)  # 1st+2nd -> load
    assert _run(0b111, 0, BB) == (0b111, 1, 0)  # bases-loaded walk scores one
    assert _run(0b100, 0, BB) == (0b101, 0, 0)  # runner on 3B not forced


def test_single_scores_from_third_always():
    nb, r, _oa = _run(0b100, 1, S1, u=[0.99, 0.99, 0.99])
    assert r == 1 and (nb & 0b001)  # 3B scored, batter on 1B


def test_single_runner_from_second_scores_with_low_uniform():
    _, r_scores, _ = _run(0b010, 0, S1, u=[0.9, 0.10, 0.9])  # u1 < 0.60 -> scores
    _, r_holds, _ = _run(0b010, 0, S1, u=[0.9, 0.95, 0.9])  # u1 > 0.60 -> to 3B
    assert r_scores == 1
    assert r_holds == 0


def test_double_and_triple():
    # double, runners on 1B+3B: 3B scores always; 1B runner holds at 3B (u1 high)
    nb, r, _ = _run(0b101, 0, S2, u=[0.9, 0.99, 0.9])
    assert (r, nb) == (1, 0b110)  # batter on 2B, ex-1B runner on 3B
    # same but u1 low -> ex-1B runner also scores
    assert _run(0b101, 0, S2, u=[0.9, 0.10, 0.9])[1] == 2
    assert _run(0b111, 0, S3) == (0b100, 3, 0)  # everyone scores, batter on 3B


def test_ball_in_play_out_can_plate_run_from_third_under_two_outs():
    _, r_sac, oa = _run(0b100, 1, FIELD_OUT, u=[0.10, 0.9, 0.9])  # u0 < 0.32
    _, r_none, _ = _run(0b100, 1, FIELD_OUT, u=[0.9, 0.9, 0.9])
    _, r_two_out, _ = _run(0b100, 2, FIELD_OUT, u=[0.01, 0.9, 0.9])  # 2 outs -> no run
    assert (r_sac, oa) == (1, 1)
    assert r_none == 0
    assert r_two_out == 0


def test_vectorised_shapes_and_dtypes():
    n = 500
    rng = np.random.default_rng(0)
    base = rng.integers(0, 8, n)
    outs = rng.integers(0, 3, n)
    event = rng.integers(0, 8, n)
    u = rng.random((n, 3))
    nb, r, oa = advance_batch(base, outs, event, u)
    assert nb.shape == r.shape == oa.shape == (n,)
    assert nb.min() >= 0 and nb.max() <= 7
    assert set(np.unique(oa)).issubset({0, 1})
    assert r.min() >= 0 and r.max() <= 4
