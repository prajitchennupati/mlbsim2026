"""Dimension tables (``warehouse`` schema): parks, teams, players, seasons."""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import ForeignKey, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from mlbsim_core.db import Base

_SCHEMA = "warehouse"


class Park(Base):
    __tablename__ = "parks"
    __table_args__ = {"schema": _SCHEMA}

    park_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(120))
    lat: Mapped[float | None]
    lon: Mapped[float | None]
    altitude_ft: Mapped[int | None]
    roof_type: Mapped[str | None] = mapped_column(String(20))  # open | fixed | retractable
    orientation_deg: Mapped[float | None]
    lf_dist: Mapped[int | None]
    cf_dist: Mapped[int | None]
    rf_dist: Mapped[int | None]
    pf_runs_3yr: Mapped[float | None]
    pf_hr_3yr: Mapped[float | None]


class Team(Base):
    __tablename__ = "teams"
    __table_args__ = {"schema": _SCHEMA}

    team_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)  # MLBAM id
    abbr: Mapped[str] = mapped_column(String(4), index=True)
    name: Mapped[str] = mapped_column(String(80))
    league: Mapped[str | None] = mapped_column(String(2))  # AL | NL
    division: Mapped[str | None] = mapped_column(String(4))  # E | C | W
    park_id: Mapped[int | None] = mapped_column(ForeignKey(f"{_SCHEMA}.parks.park_id"))


class Player(Base):
    __tablename__ = "players"
    __table_args__ = {"schema": _SCHEMA}

    player_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)  # MLBAM id
    retro_id: Mapped[str | None] = mapped_column(String(10), index=True)
    fg_id: Mapped[str | None] = mapped_column(String(12), index=True)
    bbref_id: Mapped[str | None] = mapped_column(String(12), index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    bats: Mapped[str | None] = mapped_column(String(1))  # L | R | S
    throws: Mapped[str | None] = mapped_column(String(1))  # L | R
    birth_date: Mapped[dt.date | None]
    primary_pos: Mapped[str | None] = mapped_column(String(3))
    debut_date: Mapped[dt.date | None]


class Season(Base):
    __tablename__ = "seasons"
    __table_args__ = {"schema": _SCHEMA}

    season: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=False)
    start_date: Mapped[dt.date | None]
    end_date: Mapped[dt.date | None]
    n_playoff_teams: Mapped[int | None] = mapped_column(SmallInteger)
    rules_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    notes: Mapped[str | None] = mapped_column(Text)
