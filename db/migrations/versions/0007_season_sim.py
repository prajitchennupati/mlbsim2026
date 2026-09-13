"""serving schema: season_sim_team_results, series_predictions

Revision ID: 0007_season_sim
Revises: 0006_player_predictions
Create Date: 2026-08-30
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

import mlbsim_data.models  # noqa: F401
from mlbsim_core.db import Base

revision: str = "0007_season_sim"
down_revision: str | None = "0006_player_predictions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES: tuple[str, ...] = (
    "serving.season_sim_team_results",
    "serving.series_predictions",
)


def _tables() -> list:
    return [Base.metadata.tables[name] for name in _TABLES]


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind(), tables=_tables(), checkfirst=False)


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind(), tables=_tables(), checkfirst=False)
