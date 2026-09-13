"""Performance benchmarks for the simulation engines.

Run: ``python scripts/benchmark.py`` (add ``--heavy`` for the 500k-game / 1M-season
cases). Prints a Markdown table; paste it into ``docs/PERFORMANCE.md`` with the
machine and date.
"""

from __future__ import annotations

import argparse
import platform
import time

import numpy as np

from mlbsim_engine.game import simulate_game
from mlbsim_engine.season import SeasonSetup, simulate_season

_GAMES_LEFT = 1_500  # typical "remaining schedule" size for a mid-season season MC


def _timed(fn, reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def _season_setup(seed: int = 0) -> SeasonSetup:
    rng = np.random.default_rng(seed)
    strength = rng.normal(0.0, 0.06, 30)
    home = rng.integers(0, 30, _GAMES_LEFT)
    away = (home + rng.integers(1, 30, _GAMES_LEFT)) % 30
    dp = 1.0 / (1.0 + 10.0 ** (-(strength[home] - strength[away] + 0.04)))
    return SeasonSetup(
        team_ids=list(range(30)),
        league=np.repeat([0, 1], 15),
        division=np.tile(np.repeat([0, 1, 2], 5), 2),
        wins_to_date=rng.integers(30, 60, 30),
        losses_to_date=rng.integers(30, 60, 30),
        strength=strength,
        rem_home=home,
        rem_away=away,
        rem_p_home=dp,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--heavy", action="store_true", help="also run the 500k-game / 1M-season cases")
    args = ap.parse_args()

    print(
        f"\nmachine: {platform.platform()}  ·  python {platform.python_version()}  "
        f"·  numpy {np.__version__}\n"
    )

    rows: list[tuple[str, str, float, str]] = []

    for n in [1_000, 10_000, 100_000] + ([500_000] if args.heavy else []):
        dt = _timed(lambda n=n: simulate_game(n_sims=n, seed=0, track_batting=False), args.reps)
        rows.append(("game sim (PA-level)", f"{n:,} sims", dt, f"{n / dt:,.0f} sims/s"))
    dt = _timed(lambda: simulate_game(n_sims=100_000, seed=0, track_batting=True), args.reps)
    rows.append(
        ("game sim + batting/pitcher lines", "100,000 sims", dt, f"{100_000 / dt:,.0f} sims/s")
    )

    setup = _season_setup()
    for n in [10_000, 100_000] + ([1_000_000] if args.heavy else []):
        dt = _timed(lambda n=n: simulate_season(setup, n_sims=n, seed=0), args.reps)
        decisions = n * _GAMES_LEFT
        rows.append(
            (
                f"season MC ({_GAMES_LEFT:,} games left)",
                f"{n:,} seasons",
                dt,
                f"{decisions / dt / 1e6:,.0f} M game-decisions/s",
            ),
        )

    w = max(len(r[0]) for r in rows)
    print(f"| {'benchmark':<{w}} | {'size':>14} | {'best time':>10} | {'throughput':>24} |")
    print(f"|{'-' * (w + 2)}|{'-' * 16}|{'-' * 12}|{'-' * 26}|")
    for name, size, dt, thru in rows:
        t = f"{dt * 1000:.0f} ms" if dt < 1 else f"{dt:.2f} s"
        print(f"| {name:<{w}} | {size:>14} | {t:>10} | {thru:>24} |")
    print()


if __name__ == "__main__":
    main()
