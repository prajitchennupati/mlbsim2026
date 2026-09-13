"""Transactions and rating tables (``warehouse`` schema)."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import BigInteger, ForeignKey, Index, Numeric, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from mlbsim_core.db import Base

_SCHEMA = "warehouse"


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        Index("ix_transactions_player", "player_id"),
        Index("ix_transactions_effective", "effective_date"),
        {"schema": _SCHEMA},
    )

    txn_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    player_id: Mapped[int | None] = mapped_column(ForeignKey(f"{_SCHEMA}.players.player_id"))
    team_id: Mapped[int | None] = mapped_column(ForeignKey(f"{_SCHEMA}.teams.team_id"))
    from_team_id: Mapped[int | None] = mapped_column(ForeignKey(f"{_SCHEMA}.teams.team_id"))
    type_code: Mapped[str | None] = mapped_column(String(8))  # SC TR REL CLW SFA OPT etc.
    type_desc: Mapped[str | None] = mapped_column(String(60))
    description: Mapped[str | None] = mapped_column(String(600))
    effective_date: Mapped[dt.date | None]
    resolution_date: Mapped[dt.date | None]


class EloRating(Base):
    __tablename__ = "elo_ratings"
    __table_args__ = (
        Index("ix_elo_ratings_entity", "entity_type", "entity_id", "as_of_date"),
        {"schema": _SCHEMA},
    )

    entity_type: Mapped[str] = mapped_column(String(8), primary_key=True)  # team | pitcher
    entity_id: Mapped[int] = mapped_column(primary_key=True)
    as_of_date: Mapped[dt.date] = mapped_column(primary_key=True)
    rating: Mapped[float] = mapped_column(Numeric(7, 2))
    rating_sp_adj: Mapped[float | None] = mapped_column(Numeric(7, 2))
    games_played: Mapped[int] = mapped_column(SmallInteger, default=0)
