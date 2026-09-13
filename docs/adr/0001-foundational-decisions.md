# ADR 0001 — Foundational decisions

- **Status:** accepted
- **Date:** 2026-08-30

## Context

Greenfield MLB prediction and simulation platform for the 2026 season. Several choices
fork the architecture and are expensive to reverse later.

## Decisions

### 1. Simulation-first architecture

One plate-appearance-level Monte Carlo game engine is the core. All game/player/inning
outputs are read-offs from simulated games rather than individual regressors per metric.

**Why:** ~60 target metrics that must be mutually consistent; per-metric models disagree
and multiply the leakage surface. A simulator gives internally consistent joint
distributions and makes new metrics a query. A separate fast direct path (Elo + GLM + GBM)
feeds the 100k-season Monte Carlo, where the PA engine would be intractable.

**Trade-off:** the simulator is the highest-effort component and its accuracy ceiling
depends on the event model. Mitigated by keeping the direct path as an independent
cross-check and ensembling the two.

### 2. Orchestration: GitHub Actions + cron

**Why:** free, no extra infrastructure, config-as-code visible in the repo, adequate for a
daily batch pipeline plus a game-window cron. Prefect/Dagster rejected for v1 as
operational overhead disproportionate to a single daily DAG.

**Trade-off:** weak observability and no native backfill UI. Revisit if the pipeline graph
grows complex (candidate: Dagster, for its asset lineage model).

### 3. Hosting: Render (API) + Supabase (Postgres) + Vercel (web)

**Why:** generous free/low tiers, minimal ops, clean separation of concerns, fast deploy
loop. Supabase adds a usable Postgres dashboard.

**Trade-off:** less "infrastructure" resume signal than AWS. Acceptable — the project's
signal is in the ML and simulation work, not in hand-rolled infra.

### 4. Historical data depth: 2015+ full, 2010+ games only

**Why:** Statcast begins in 2015, so a consistent single-regime event model wants
2015-onward PA + Statcast. Games and team logs back to 2010 give Elo and W/L baselines a
longer runway without forcing the event model to straddle a pre-Statcast feature regime.

**Trade-off:** shorter history for rare-event calibration. Acceptable for v1; extending
back to 2000 (games/PA without Statcast) is a possible later addition.

### 5. Live in-game predictions deferred to M9

**Why:** pre-game predictions must be proven end-to-end first; live adds latency-sensitive
infrastructure and a separate state model that would fragment focus during the core
modelling milestones. Game pages still show full pre-game predictions from M8.

**Trade-off:** the flashiest feature arrives late. Acceptable — it is additive and lands
before the bulk of the 2026 season.

## Consequences

- The `mlbsim-engine` package is on the critical path and gets a CI performance gate from
  M3.
- All prediction storage is immutable and model-versioned from M2 so that a live-season
  retrain cadence never orphans history.
- No Kubernetes, no Terraform, no message queue in v1.
