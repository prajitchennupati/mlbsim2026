"""prediction_outcomes.pred_id ON DELETE CASCADE

Revision ID: 0010_prediction_outcomes_cascade
Revises: 0009_jsonb_server_defaults
Create Date: 2026-08-31

Rebuild pipelines (``build_elo``, ``predict_date`` …) delete and re-insert their
``game_predictions`` rows. Once a prediction has been scored, the old NO ACTION
FK from ``prediction_outcomes`` blocked that delete. Cascade the stale outcome
away instead — it is re-derived on the next resolve pass.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0010_prediction_outcomes_cascade"
down_revision: str | None = "0009_jsonb_server_defaults"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONSTRAINT = "fk_prediction_outcomes_pred_id_game_predictions"
_TABLE = "prediction_outcomes"
_REFERRED = "game_predictions"


def upgrade() -> None:
    op.drop_constraint(_CONSTRAINT, _TABLE, schema="serving", type_="foreignkey")
    op.create_foreign_key(
        _CONSTRAINT,
        _TABLE,
        _REFERRED,
        ["pred_id"],
        ["pred_id"],
        source_schema="serving",
        referent_schema="serving",
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint(_CONSTRAINT, _TABLE, schema="serving", type_="foreignkey")
    op.create_foreign_key(
        _CONSTRAINT,
        _TABLE,
        _REFERRED,
        ["pred_id"],
        ["pred_id"],
        source_schema="serving",
        referent_schema="serving",
    )
