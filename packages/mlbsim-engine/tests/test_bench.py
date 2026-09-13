"""Performance gate for the game simulator (CI enforces the wall-clock)."""

from __future__ import annotations

import time

import pytest

from mlbsim_engine.game import simulate_game

# Generous ceiling for a cold CI runner; the local number is ~0.6 s.
_BUDGET_SECONDS_10K = 4.0


@pytest.mark.bench
def test_10k_sims_under_budget():
    start = time.perf_counter()
    simulate_game(n_sims=10_000, seed=0, track_batting=True)
    elapsed = time.perf_counter() - start
    assert elapsed < _BUDGET_SECONDS_10K, (
        f"10k sims took {elapsed:.2f}s (budget {_BUDGET_SECONDS_10K}s)"
    )


@pytest.mark.bench
def test_throughput_scales_roughly_linearly():
    def timed(n: int) -> float:
        start = time.perf_counter()
        simulate_game(n_sims=n, seed=1, track_batting=False)
        return time.perf_counter() - start

    t_small = timed(4_000)
    t_big = timed(16_000)
    # 4x the work should not be more than ~8x the time (loose bound; guards regressions)
    assert t_big < 8 * t_small + 0.5
