"""Point-in-time feature store (``warehouse`` schema).

Every ``game_features`` row is stamped with the ``feature_set`` version and a
``data_snapshot_hash`` (hash of the source rows that fed it), so a recomputation
that silently pulls in future data changes the hash and fails CI.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import BigInteger, ForeignKey, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from mlbsim_core.db import Base
from mlbsim_data.models._types import ingested_ts, ts

_SCHEMA = "warehouse"


class FeatureSet(Base):
    __tablename__ = "feature_sets"
    __table_args__ = {"schema": _SCHEMA}

    feature_set_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # e.g. "fs_v1"
    name: Mapped[str] = mapped_column(String(80))
    version: Mapped[str] = mapped_column(String(16))
    spec_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[ingested_ts]


class GameFeature(Base):
    __tablename__ = "game_features"
    __table_args__ = {"schema": _SCHEMA}

    game_pk: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{_SCHEMA}.games.game_pk"), primary_key=True
    )
    feature_set_id: Mapped[str] = mapped_column(
        String(64), ForeignKey(f"{_SCHEMA}.feature_sets.feature_set_id"), primary_key=True
    )
    side: Mapped[str] = mapped_column(String(4), primary_key=True)  # home | away | diff
    as_of_ts: Mapped[ts]
    features: Mapped[dict[str, Any]] = mapped_column(JSONB)
    data_snapshot_hash: Mapped[str] = mapped_column(String(64))
    computed_at: Mapped[ingested_ts]
