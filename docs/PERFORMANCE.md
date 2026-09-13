# Performance

Numbers are from `python scripts/benchmark.py --heavy` (`make bench`). Reproduce and
update the table with your machine + date when the engine changes.

## Latest run

- **machine:** Apple Silicon (macOS, arm64), Python 3.13, NumPy 2.5
- **date:** 2026-08-31

| benchmark | size | best time | throughput |
|---|---|---|---|
| game sim (PA-level) | 1,000 sims | 57 ms | ~18k sims/s |
| game sim (PA-level) | 10,000 sims | 0.53 s | ~19k sims/s |
| game sim (PA-level) | 100,000 sims | 5.9 s | ~17k sims/s |
| game sim (PA-level) | 500,000 sims | 31.3 s | ~16k sims/s |
| game sim + batting/pitcher lines | 100,000 sims | 6.8 s | ~15k sims/s |
| season MC (1,500 games left) | 10,000 seasons | 89 ms | ~169 M game-decisions/s |
| season MC (1,500 games left) | 100,000 seasons | 0.88 s | ~170 M game-decisions/s |
| season MC (1,500 games left) | 1,000,000 seasons | 8.8 s | ~170 M game-decisions/s |

## What each engine is for

**Game simulator (`mlbsim_engine.game`)** — plate-appearance level, one matchup at a time.
The product use is the game page and the live win-probability endpoint: **10,000 sims in
~0.55 s**, which is the size those actually request. It is also used to *calibrate* the
fast direct path. It is **not** used to drive the season Monte Carlo (see ADR 0001).

**Season Monte Carlo (`mlbsim_engine.season`)** — draws every remaining game from a
supplied per-game win probability (the fast direct path), accumulates standings, seeds
each league, and runs the bracket. 100k seasons × 1,500 remaining games ≈ 150 M game
decisions in under a second; 1 M seasons in ~9 s. This is the engine behind every
playoff / pennant / World-Series probability.

## Scaling

`simulate_game` processes `n_sims` in cohorts of `_CHUNK = 25,000` (`_simulate_cohort`,
seeded per chunk from `np.random.default_rng([seed, chunk_index])`) and concatenates.
Before chunking, throughput fell off super-linearly (100k sims took 7 s, 1 M took ~95 s)
because each of the ~150 per-PA iterations allocated several `(n,)` scratch arrays and the
working set blew the cache at large `n`. With chunking the working set is bounded and
throughput is flat in `n_sims` (100k in 5.9 s, 500k in 31 s → linear; 1 M ≈ 63 s).

`simulate_season` was already chunked (5,000 seasons/batch with online standings
aggregation) from M5.

## Why not Numba / a compiled kernel

The game engine's hot path is a handful of **NumPy array operations per plate appearance**
over `(n, 8)` matrices — an odds-ratio matchup blend, an inverse-CDF multinomial draw, and
a branchless vectorised baserunning update — repeated ~150 times per simulated game. There
is no Python-level loop over sims or over plate appearances for Numba to compile away; the
interpreter is already out of the inner loop.

A materially faster engine would need a *fused* per-PA kernel (a scalar loop over one
game's PAs, JIT-compiled or written in Rust/Cython, then parallelised across sims) — a
rewrite of `game.py`, `rates.py`, and `baserunning.py` with a second code path to
maintain. At current throughput (10k sims / matchup in ~0.55 s, 1 M-season MC in ~9 s)
nothing in the platform needs it: the game page, the live endpoint, and the nightly
recalibration all run comfortably. It is recorded as a deferred option, not a gap.

The `mlbsim-engine[jit]` extra (`numba`) is kept as a stub for a contributor who wants to
prototype that kernel and benchmark it against the vectorised baseline.

## CI gate

`packages/mlbsim-engine/tests/test_bench.py` (marker `bench`, runs in the default suite):
10k sims must finish under 4 s on a cold CI runner (local ~0.55 s), and 4× the work must
not take more than ~8× the time (guards against a scaling regression).
