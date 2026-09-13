# ADR 0002 — Vectorised cohort simulation, chunked; no Numba

- **Status:** accepted
- **Date:** 2026-08-31 (decided M3, confirmed + measured M10)

## Context

The plate-appearance engine runs tens of thousands of simulated games for the game page
and the live win-probability endpoint. The obvious design is a per-game Python loop over
plate appearances, JIT-compiled with Numba; `docs/ROADMAP.md` listed a Numba pass as an
M10 optimisation.

## Decision

**1. Simulate the whole cohort at once.** All `n` games advance together, one plate
appearance per iteration for the sims currently batting, every step expressed as NumPy
array ops over `(n,)` / `(n, 8)` arrays (odds-ratio matchup blend, inverse-CDF multinomial
draw, branchless vectorised baserunning, masked inning/half transitions). No per-game
Python loop.

**2. Chunk large `n`.** `simulate_game` processes `n_sims` in cohorts of `_CHUNK = 25,000`
(`_simulate_cohort`, seeded per chunk from `np.random.default_rng([seed, chunk_index])`)
and concatenates the result arrays. The season Monte Carlo was already chunked from M5.

**3. Do not adopt Numba.** The hot path is NumPy array calls, not a Python loop — there is
nothing for a JIT to compile away. See "Consequences".

## Consequences (measured M10, Apple Silicon, see `docs/PERFORMANCE.md`)

- **Game engine:** 10,000 sims / matchup in ~0.55 s (~18k sims/s) — the size the game
  page and live endpoint actually request. Throughput is flat in `n_sims`.
- **Chunking removed a super-linear cliff:** before it, 100k sims took ~7 s and 1 M took
  ~95 s (each of ~150 per-PA iterations allocated several `(n,)` scratch arrays; at large
  `n` the working set left cache). After: 100k in 6.3 s, 500k in 32 s — linear.
- **Season MC:** 100k seasons ≈ 0.9 s, 1 M ≈ 9 s, ~167 M game-decisions/s. Untouched.
- **Numba not adopted.** A materially faster engine needs a *fused* per-PA scalar kernel
  (JIT or Rust/Cython, parallelised across sims) — a rewrite of `game.py` / `rates.py` /
  `baserunning.py` with a second code path. Nothing in the platform needs it at current
  throughput. The `mlbsim-engine[jit]` extra stays as a stub for a future benchmark.
- The "no per-row Python" constraint is load-bearing: `baserunning.advance_batch` and the
  inning-transition block are branchless masked updates on purpose; `np.add.at` was
  replaced with plain `arr[idx] += v` (indices are unique per iteration). `test_bench.py`
  gates the wall-clock and the scaling.

## Trade-off

Cohort code is harder to read than a straight per-PA loop, and one very deep extra-inning
game makes its whole chunk take an extra masked iteration. Both are acceptable given the
throughput and the CI gate.
