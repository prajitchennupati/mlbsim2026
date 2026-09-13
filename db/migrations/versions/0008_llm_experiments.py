"""serving schema: llm_experiments

Revision ID: 0008_llm_experiments
Revises: 0007_season_sim
Create Date: 2026-08-30
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

import mlbsim_data.models  # noqa: F401
from mlbsim_core.db import Base

revision: str = "0008_llm_experiments"
down_revision: str | None = "0007_season_sim"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES: tuple[str, ...] = ("serving.llm_experiments",)


def _tables() -> list:
    return [Base.metadata.tables[name] for name in _TABLES]


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind(), tables=_tables(), checkfirst=False)


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind(), tables=_tables(), checkfirst=False)
