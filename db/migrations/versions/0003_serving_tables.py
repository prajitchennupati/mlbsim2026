"""serving schema tables: model_versions, game_predictions, prediction_outcomes

Revision ID: 0003_serving_tables
Revises: 0002_warehouse_tables
Create Date: 2026-08-30
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

import mlbsim_data.models  # noqa: F401  -- registers tables on Base.metadata
from mlbsim_core.db import Base

revision: str = "0003_serving_tables"
down_revision: str | None = "0002_warehouse_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES: tuple[str, ...] = (
    "serving.model_versions",
    "serving.game_predictions",
    "serving.prediction_outcomes",
    "serving.model_eval_runs",
    "serving.calibration_bins",
)


def _tables() -> list:
    return [Base.metadata.tables[name] for name in _TABLES]


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind(), tables=_tables(), checkfirst=False)


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind(), tables=_tables(), checkfirst=False)
