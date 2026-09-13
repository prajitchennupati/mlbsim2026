# Contributing

## Prerequisites

- Python 3.12+ (CI pins 3.12)
- [`uv`](https://docs.astral.sh/uv/) — preferred. The `Makefile` falls back to a
  pip-managed `.venv` if `uv` is absent.
- Docker (for local Postgres + Redis)
- Node 20+ (only for `web/`, from M8)

## Setup

```bash
cp .env.example .env
make setup          # uv sync (or create .venv and pip install -e all packages)
make up             # start Postgres + Redis
make migrate        # apply Alembic migrations
make check          # lint + typecheck + tests — must pass before every commit
```

## Layout

`uv` workspace monorepo. Each package under `packages/` is independently
installable and owns its own tests. A package may import another package's public
API only (its top-level module), never its internals. `notebooks/` is never
imported by package code.

| Package | Responsibility |
|---|---|
| `mlbsim-core` | settings, DB session, logging |
| `mlbsim-data` | ingestion, validation, warehouse loaders, ID crosswalk |
| `mlbsim-features` | point-in-time feature engineering |
| `mlbsim-models` | ratings, models, projections, ensemble, explainability |
| `mlbsim-engine` | game simulator, season Monte Carlo, playoff bracket |
| `mlbsim-pipeline` | orchestration flows + the `mlbsim` CLI |
| `mlbsim-api` | read-only FastAPI service |

## Conventions

- `ruff` for lint + format, `mypy --strict` for types. Both gate CI.
- Tests: `pytest`. Mark tests needing a live database `@pytest.mark.integration`
  and those hitting a network `@pytest.mark.network`; `make test` skips
  integration.
- Migrations: edit ORM models in `mlbsim_data.models`, then
  `uv run alembic -c db/alembic.ini revision --autogenerate -m "…"`. Review the
  generated file. `make sql` previews DDL without a database.
- Commits: conventional-commit style (`feat:`, `fix:`, `chore:`, `docs:`).
  Keep `main` green.
- Every model added ships an entry in `serving.model_versions` and a metrics row
  against its named baseline (see `docs/EVALUATION.md`).

## Milestones

See [`docs/ROADMAP.md`](docs/ROADMAP.md). Each milestone must leave the system
runnable and CI green.
