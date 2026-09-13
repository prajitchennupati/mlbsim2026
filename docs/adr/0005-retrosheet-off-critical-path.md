# ADR 0005 — Retrosheet dropped from the ingestion critical path

- **Status:** accepted
- **Date:** 2026-08-31 (decided M1)

## Context

`docs/DATA_SOURCES.md` and ADR 0001 list Retrosheet as a play-by-play source. Retrosheet
is authoritative and goes back decades, but its event files need a separate parser
(`.EVN`/`.EVA` format, the Chadwick `cwevent` toolchain) and a separate ID crosswalk
step, and it lags the current season by months.

## Decision

The backfill window is 2015+ (ADR 0001 §4). Over that window the MLB Stats API GUMBO feed
(`/v1.1/game/{pk}/feed/live`, `allPlays`) already provides every plate appearance with
base-out state, pitch-level data, and lineups — which the M1 parser reconstructs and
validates against real fixtures. Retrosheet would duplicate that for the window we
actually use.

Retrosheet is therefore **not on the critical path**. It stays in `DATA_SOURCES.md` as an
optional future source for a pre-2015 extension (games/PA without Statcast, per ADR 0001's
trade-off note), behind its own parser package if and when that extension happens.

## Consequences

- One ingestion path (Stats API) for the whole modelling window; `mlbsim-data` has no
  Retrosheet dependency.
- The Chadwick Bureau register is still used, but only for the player-ID crosswalk
  (`retro_id` / `fg_id` / `bbref_id` columns), not for events.
- If pre-2015 history is added later, it is a new source package + migration, not a
  change to the existing pipeline.

## Trade-off

We trust MLB's own play classification for 2015+ rather than cross-checking it against
Retrosheet. Acceptable — the fixtures-based parser tests catch structural regressions,
and the event model is robust to small classification noise.
