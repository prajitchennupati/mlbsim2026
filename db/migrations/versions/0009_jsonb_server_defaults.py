"""server defaults for JSONB "bag" columns

Revision ID: 0009_jsonb_server_defaults
Revises: 0008_llm_experiments
Create Date: 2026-08-31

Several JSONB columns are ``NOT NULL`` but only had a Python-side ``default=dict``,
so any raw-SQL / Core insert that omitted them failed. Add a matching
``DEFAULT '{}'::jsonb`` so every write path is safe.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0009_jsonb_server_defaults"
down_revision: str | None = "0008_llm_experiments"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS: tuple[tuple[str, str], ...] = (
    ("serving.model_versions", "hyperparams_json"),
    ("serving.model_versions", "metrics_json"),
    ("serving.simulation_runs", "params_json"),
    ("serving.llm_experiments", "train_spec_json"),
    ("serving.llm_experiments", "metrics_json"),
    ("warehouse.feature_sets", "spec_json"),
)


def upgrade() -> None:
    for table, column in _COLUMNS:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} SET DEFAULT '{{}}'::jsonb")
        op.execute(f"UPDATE {table} SET {column} = '{{}}'::jsonb WHERE {column} IS NULL")


def downgrade() -> None:
    for table, column in _COLUMNS:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} DROP DEFAULT")
