# MLB Playoffs 2026 — AI Prediction & World Series Simulation Platform

A production-quality MLB analytics platform that forecasts every meaningful aspect of the
2026 MLB season — game outcomes, final scores, runs by inning, player stat lines, pitcher
strikeout distributions, playoff odds, and World Series probabilities — using machine
learning, probabilistic modelling, and plate-appearance-level Monte Carlo simulation.

> **Status:** milestones M0–M9 built (design → ingestion → models → simulation → API →
> frontend → automation). The pipeline, model, and API layers are verified end-to-end
> against a live PostgreSQL — `mlbsim flow daily` runs green on real MLB data. Not yet
> deployed; the frontend needs a `next build` on a Node box. See
> [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## What it predicts

- **Game level** — win probability, predicted score, full score distribution, expected
  total runs, run differential, P(shutout), P(one-run game), P(extra innings), runs by
  inning with P(team scores 1+/2+/3+), P(lead after each inning), comeback probability.
- **Player level** — projected PA / AB / H / 2B / 3B / HR / R / RBI / BB / K / HBP / TB
  per batter, plus P(1+ hit), P(2+ hits), P(HR), P(RBI), P(run), P(walk), P(K) as full
  distributions rather than point estimates.
- **Pitcher level** — IP, hits, ER, BB, K, HR allowed, pitch count, batters faced, WHIP,
  game ERA, quality-start probability, plus P(5+/6+/7+/8+/10+ K) and P(5+/6+/7+ IP).
- **Season** — division / wild-card / playoff / pennant / World Series probability,
  expected wins–losses, expected seed, from 100,000 simulated remaining seasons.
- **Postseason** — interactive bracket, per-series win probability, expected series
  length, P(sweep / 5 / 6 / 7 games), World Series matchup and winner odds.

## Core idea — simulation first

Rather than train a separate regressor for each of ~60 target metrics (which would be
mutually inconsistent and a leakage minefield), the system centres on **one
plate-appearance-level Monte Carlo game engine**. Every downstream number is a read-off
from simulated games, so all outputs are internally consistent and come with full
distributions. The engine is parameterised by four learned components: a PA-outcome
event model, a baserunning transition model, a bullpen-usage model, and player talent
projections. A fast "direct" path (Elo + logistic + Poisson/NB + GBM) feeds the
100k-season Monte Carlo, where running the full PA engine would be intractable.

See [`docs/MODELING.md`](docs/MODELING.md) for the full rationale.

## Stack

| Layer | Technology |
|---|---|
| Data | MLB Stats API (GUMBO), Baseball Savant / Statcast, Chadwick register, Open-Meteo |
| Warehouse | PostgreSQL (Supabase — via the **session pooler**, see [ADR 0006](docs/adr/0006-supabase-session-pooler.md)) |
| ML | scikit-learn, statsmodels, exact linear SHAP (no `shap` dep), optional MLflow + Anthropic |
| Simulation | pure vectorised NumPy — cohort PA engine + chunked season Monte Carlo ([ADR 0002](docs/adr/0002-vectorised-cohort-simulation.md)) |
| Backend | FastAPI, Redis-or-in-process cache, Pydantic, Alembic |
| Frontend | Next.js (App Router), TypeScript, Tailwind CSS, Recharts |
| Orchestration | GitHub Actions — `mlbsim flow daily` / `flow live` / `nightly-train` |
| Deployment | Render (API), Vercel (web), Supabase (Postgres) |

## Repository layout

```
packages/
  mlbsim-data/       ingestion, validation, warehouse loaders, ID crosswalk
  mlbsim-features/   point-in-time feature store (AS-OF joins only)
  mlbsim-models/     ratings, event model, direct models, projections, ensemble, explain, llm, evaluate
  mlbsim-engine/     PA-level game simulator, season Monte Carlo, playoff bracket
  mlbsim-pipeline/   orchestration flows + CLI
  mlbsim-api/        FastAPI service
web/                 Next.js frontend
db/migrations/       Alembic
docs/                architecture, database, modelling, evaluation, roadmap, ADRs
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system architecture, data flow, hard problems
- [`docs/DATABASE.md`](docs/DATABASE.md) — schema, tables, indexing, partitioning
- [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) — sources, backfill scope, ingestion contract, attribution
- [`docs/MODELING.md`](docs/MODELING.md) — models, rationale, data-leakage register
- [`docs/LLM_EXPERIMENT.md`](docs/LLM_EXPERIMENT.md) — the honest "does an LLM beat the GBM?" test
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — validation protocol, metrics, baselines, results
- [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — engine benchmarks and scaling
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — phased implementation plan
- [`docs/adr/`](docs/adr/) — architecture decision records (0001–0006)

## Local development

`uv` is preferred; `make setup` falls back to a plain `.venv` + pip when it is absent.
A PostgreSQL 16+ is needed for the pipeline, the API against real data, and the
integration tests — a local container or a Supabase project (use the **session-pooler**
connection string, port 5432).

```bash
make setup                       # install the workspace + dev tools
export MLBSIM_DATABASE_URL=postgresql+psycopg://…@…pooler.supabase.com:5432/postgres?sslmode=require
make migrate                     # alembic upgrade head  (mlbsim db upgrade)

mlbsim ingest schedule 2024-07-01 2024-07-14
mlbsim ingest game <gamePk> --final
mlbsim flow daily --date 2024-07-13 --slate 2024-07-14 --no-publish
mlbsim flow status

make test        # unit suite (no DB)
make test-int    # integration suite (needs the DB)
make coverage    # combined unit + integration coverage
make bench       # engine benchmarks -> docs/PERFORMANCE.md
make check       # everything CI's quality job runs
```

The `mlbsim` CLI is the single entry point (`mlbsim --help`): `db`, `ingest`, `transform`,
`features`, `elo`, `train`, `predict`, `ensemble`, `simulate-game`, `simulate-season`,
`evaluate`, `explain`, `flow daily|live|status`.

## Data & attribution

Non-commercial research / portfolio project. Data from the MLB Stats API, Baseball Savant
(Statcast), Retrosheet, FanGraphs, Baseball-Reference, the Chadwick Bureau register, and
Open-Meteo. See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) §5 for terms of use.

> The information used here was obtained free of charge from and is copyrighted by
> Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.

## Licence

MIT (code only; data belongs to its respective providers under their terms)
