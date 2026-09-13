"""warehouse schema tables (dims, game facts, play-by-play, stats, transactions, elo)

Revision ID: 0002_warehouse_tables
Revises: 0001_create_schemas
Create Date: 2026-08-30

Baselines the full ``warehouse`` schema from ``mlbsim_data.models``. The table
set is frozen as ``_TABLES`` so this migration stays a fixed snapshot even as the
ORM models grow in later milestones (each later change gets its own migration).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

import mlbsim_data.models  # noqa: F401  -- registers tables on Base.metadata
from mlbsim_core.db import Base

revision: str = "0002_warehouse_tables"
down_revision: str | None = "0001_create_schemas"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen snapshot of the warehouse tables introduced at M1.
_TABLES: tuple[str, ...] = (
    "warehouse.parks",
    "warehouse.teams",
    "warehouse.players",
    "warehouse.seasons",
    "warehouse.games",
    "warehouse.game_weather",
    "warehouse.game_probables",
    "warehouse.lineups",
    "warehouse.plate_appearances",
    "warehouse.pitches",
    "warehouse.batted_balls",
    "warehouse.batting_game_logs",
    "warehouse.pitching_game_logs",
    "warehouse.team_game_logs",
    "warehouse.player_season_stats",
    "warehouse.team_season_stats",
    "warehouse.transactions",
    "warehouse.elo_ratings",
)


def _tables() -> list:
    return [Base.metadata.tables[name] for name in _TABLES]


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind, tables=_tables(), checkfirst=False)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind, tables=_tables(), checkfirst=False)
