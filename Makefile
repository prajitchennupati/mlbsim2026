# mlbplayoffs2026 developer tasks.
#
# Uses `uv` when available (preferred); otherwise falls back to a local .venv
# managed with pip. Run `make setup` first.

SHELL := /bin/bash
UV := $(shell command -v uv 2>/dev/null)

ifdef UV
  RUN := uv run
  SETUP_CMD := uv sync --all-extras --all-packages
else
  RUN := .venv/bin/
  SETUP_CMD := python3 -m venv .venv && \
    .venv/bin/python -m pip install -U pip && \
    .venv/bin/python -m pip install -e packages/mlbsim-core -e packages/mlbsim-data \
      -e packages/mlbsim-features -e packages/mlbsim-models -e packages/mlbsim-engine \
      -e packages/mlbsim-pipeline -e packages/mlbsim-api \
      ruff mypy pytest pytest-cov httpx types-requests
endif

.DEFAULT_GOAL := help
.PHONY: help setup up down logs migrate sql current crosswalk api test test-int coverage bench lint typecheck fmt check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install all workspace packages + dev tools
	$(SETUP_CMD)

up: ## Start Postgres + Redis
	docker compose up -d
	@echo "waiting for postgres..." && \
	  until docker compose exec -T postgres pg_isready -U mlbsim -d mlbsim >/dev/null 2>&1; \
	  do sleep 1; done && echo "ready."

down: ## Stop containers (keep data)
	docker compose down

logs: ## Tail container logs
	docker compose logs -f

migrate: ## Apply Alembic migrations
	$(RUN)mlbsim db upgrade

current: ## Show current DB revision
	$(RUN)mlbsim db current

sql: ## Print the migration SQL without touching a database
	$(RUN)alembic -c db/alembic.ini upgrade head --sql

crosswalk: ## Load the Chadwick player-ID crosswalk
	$(RUN)mlbsim crosswalk

api: ## Run the API locally with reload
	$(RUN)uvicorn mlbsim_api.main:app --reload --port 8000

test: ## Run the test suite (skips integration tests)
	$(RUN)pytest -m "not integration"

test-int: ## Run only the integration tests (needs MLBSIM_DATABASE_URL)
	$(RUN)pytest -m integration

coverage: ## Combined unit + integration coverage, 85% gate (needs a live Postgres)
	$(RUN)coverage erase
	$(RUN)pytest -m "not integration" --cov --cov-report= -q
	$(RUN)pytest -m integration --cov --cov-append --cov-report= -q
	$(RUN)coverage report --fail-under=85
	$(RUN)coverage html

bench: ## Run the performance benchmarks and refresh docs/PERFORMANCE.md numbers
	$(RUN)pytest -m bench -q
	$(RUN)python scripts/benchmark.py

lint: ## Ruff lint
	$(RUN)ruff check .

fmt: ## Ruff format + autofix
	$(RUN)ruff format . && $(RUN)ruff check --fix .

typecheck: ## mypy (strict)
	$(RUN)mypy packages

check: lint typecheck test ## Everything CI runs

clean: ## Remove caches and the local venv
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache **/__pycache__ **/*.egg-info
