# ADR 0004 — One blessed as-of join for every feature

- **Status:** accepted
- **Date:** 2026-08-31 (decided M2)

## Context

Data leakage is the single biggest threat to an honest backtest. Every feature the models
see for a game on date `d` must be computable from information available strictly before
`d` — rolling team form, pitcher form, Elo, season-to-date rates. It is easy to get this
subtly wrong (a `<=` where a `<` belongs, a season-stat row that silently includes the
game being predicted).

## Decision

All point-in-time lookups go through one module, `mlbsim_features/asof.py`:

- `before(df, date)` / `asof_lookup` / `asof_join` enforce a strict `<` bound — never
  `<=` — against the observation timestamp.
- `player_season_stats` / `team_season_stats` carry an explicit `through_date`; the
  builder requests the row `through_date < game_date` (or the full-season row only when
  building for analysis, never for training).
- `snapshot_hash(...)` hashes the exact input frame; the value is written to every
  `game_features` row as `data_snapshot_hash`, so a feature set is reproducible and a
  leak shows up as a hash mismatch.
- A property test (`test asof`) asserts that no row returned by the join has an
  observation date `>=` the query date, on randomised inputs.

Nothing else in the codebase is allowed to hand-roll a "latest value as of" query.

## Consequences

- The backtest harness instantiates, per date, the model version and feature snapshot
  that would have existed then; `docs/MODELING.md` §8 keeps the leakage register.
- Adding a feature means adding it to `build.py` and letting the as-of machinery and the
  snapshot hash cover it — no per-feature leakage review.

## Trade-off

Every feature computation pays for a sorted merge / searchsorted rather than a plain
join, and the `through_date` discipline means `player_season_stats` has many more rows
than a single full-season table would. Both are cheap relative to a blown backtest.
