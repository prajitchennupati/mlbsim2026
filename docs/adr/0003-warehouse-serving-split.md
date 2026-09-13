# ADR 0003 — Two schemas: `warehouse` (facts) and `serving` (predictions)

- **Status:** accepted
- **Date:** 2026-08-31 (decided M1–M2)

## Context

The database holds two very different kinds of data: ingested facts about what happened
(games, plate appearances, Statcast, game logs, season stats) and things the platform
produced (predictions, simulation runs, evaluation snapshots, model registry). They have
different write patterns, retention, and trust levels.

## Decision

One PostgreSQL database, two schemas:

- **`warehouse`** — ingested + derived facts. Upsert-on-conflict from the loaders; the
  MLB Stats API / Baseball Savant are the source of truth. Point-in-time aggregates
  (`player_season_stats` with a `through_date`) live here too.
- **`serving`** — everything the models write: `model_versions`, `game_predictions`
  (append-only, immutable, model-versioned), `prediction_outcomes`, `simulation_runs`,
  `game_sim_results`, `player_predictions`, `season_sim_team_results`,
  `series_predictions`, `model_eval_runs`, `calibration_bins`, `llm_experiments`.

The API reads `serving` (plus team/game dimension rows) and never writes. The pipeline
writes both. Alembic manages only these two schemas (`env.py` `include_name` filters out
everything else, so the DB can be shared with Supabase's own `auth` / `storage` schemas).

## Consequences

- A retrain never mutates history — a new prediction is a new row with its own
  `model_id` + `created_at`. `prediction_outcomes.pred_id` is `ON DELETE CASCADE`
  (ADR-adjacent fix in `06258aa`) so rebuild-style pipelines can still delete-and-reinsert
  a model's predictions.
- Backups / retention / access can be reasoned about per schema.
- `serving` can be dropped and rebuilt from `warehouse` + model artifacts without
  re-ingesting anything.

## Trade-off

Cross-schema foreign keys (`serving.game_predictions.game_pk` → `warehouse.games`) need
explicit schema-qualified names everywhere and make a few queries wordier. Accepted for
the separation of concerns.
