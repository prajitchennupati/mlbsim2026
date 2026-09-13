"""serving schema: simulation_runs, game_sim_results

Revision ID: 0005_simulation_results
Revises: 0004_feature_store
Create Date: 2026-08-30
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

import mlbsim_data.models  # noqa: F401
from mlbsim_core.db import Base

revision: str = "0005_simulation_results"
down_revision: str | None = "0004_feature_store"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES: tuple[str, ...] = ("serving.simulation_runs", "serving.game_sim_results")


def _tables() -> list:
    return [Base.metadata.tables[name] for name in _TABLES]


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind(), tables=_tables(), checkfirst=False)


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind(), tables=_tables(), checkfirst=False)
