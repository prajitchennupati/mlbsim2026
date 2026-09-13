"""Aggregate stat tables (``warehouse`` schema).

Game logs are derived from boxscores (and reconcilable against
``plate_appearances``). Season stats are rolled up from game logs with a
``split`` dimension.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import BigInteger, Boolean, ForeignKey, Index, SmallInteger, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from mlbsim_core.db import Base

_SCHEMA = "warehouse"


class BattingGameLog(Base):
    __tablename__ = "batting_game_logs"
    __table_args__ = (
        Index("ix_batting_game_logs_player", "player_id"),
        {"schema": _SCHEMA},
    )

    player_id: Mapped[int] = mapped_column(
        ForeignKey(f"{_SCHEMA}.players.player_id"), primary_key=True
    )
    game_pk: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{_SCHEMA}.games.game_pk"), primary_key=True
    )
    team_id: Mapped[int] = mapped_column(ForeignKey(f"{_SCHEMA}.teams.team_id"))
    season: Mapped[int] = mapped_column(SmallInteger)
    batting_order: Mapped[int | None] = mapped_column(SmallInteger)
    pa: Mapped[int] = mapped_column(SmallInteger, default=0)
    ab: Mapped[int] = mapped_column(SmallInteger, default=0)
    r: Mapped[int] = mapped_column(SmallInteger, default=0)
    h: Mapped[int] = mapped_column(SmallInteger, default=0)
    d2b: Mapped[int] = mapped_column(SmallInteger, default=0)
    t3b: Mapped[int] = mapped_column(SmallInteger, default=0)
    hr: Mapped[int] = mapped_column(SmallInteger, default=0)
    rbi: Mapped[int] = mapped_column(SmallInteger, default=0)
    bb: Mapped[int] = mapped_column(SmallInteger, default=0)
    ibb: Mapped[int] = mapped_column(SmallInteger, default=0)
    hbp: Mapped[int] = mapped_column(SmallInteger, default=0)
    so: Mapped[int] = mapped_column(SmallInteger, default=0)
    sb: Mapped[int] = mapped_column(SmallInteger, default=0)
    cs: Mapped[int] = mapped_column(SmallInteger, default=0)
    sac: Mapped[int] = mapped_column(SmallInteger, default=0)
    sf: Mapped[int] = mapped_column(SmallInteger, default=0)
    gidp: Mapped[int] = mapped_column(SmallInteger, default=0)
    tb: Mapped[int] = mapped_column(SmallInteger, default=0)


class PitchingGameLog(Base):
    __tablename__ = "pitching_game_logs"
    __table_args__ = (
        Index("ix_pitching_game_logs_player", "player_id"),
        {"schema": _SCHEMA},
    )

    player_id: Mapped[int] = mapped_column(
        ForeignKey(f"{_SCHEMA}.players.player_id"), primary_key=True
    )
    game_pk: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{_SCHEMA}.games.game_pk"), primary_key=True
    )
    team_id: Mapped[int] = mapped_column(ForeignKey(f"{_SCHEMA}.teams.team_id"))
    season: Mapped[int] = mapped_column(SmallInteger)
    is_start: Mapped[bool] = mapped_column(Boolean, default=False)
    bf: Mapped[int] = mapped_column(SmallInteger, default=0)
    outs: Mapped[int] = mapped_column(SmallInteger, default=0)  # innings pitched * 3
    h: Mapped[int] = mapped_column(SmallInteger, default=0)
    r: Mapped[int] = mapped_column(SmallInteger, default=0)
    er: Mapped[int] = mapped_column(SmallInteger, default=0)
    bb: Mapped[int] = mapped_column(SmallInteger, default=0)
    ibb: Mapped[int] = mapped_column(SmallInteger, default=0)
    hbp: Mapped[int] = mapped_column(SmallInteger, default=0)
    so: Mapped[int] = mapped_column(SmallInteger, default=0)
    hr: Mapped[int] = mapped_column(SmallInteger, default=0)
    pitches: Mapped[int] = mapped_column(SmallInteger, default=0)
    strikes: Mapped[int] = mapped_column(SmallInteger, default=0)
    got_win: Mapped[bool] = mapped_column(Boolean, default=False)
    got_loss: Mapped[bool] = mapped_column(Boolean, default=False)
    got_save: Mapped[bool] = mapped_column(Boolean, default=False)
    quality_start: Mapped[bool] = mapped_column(Boolean, default=False)


class TeamGameLog(Base):
    __tablename__ = "team_game_logs"
    __table_args__ = {"schema": _SCHEMA}

    team_id: Mapped[int] = mapped_column(ForeignKey(f"{_SCHEMA}.teams.team_id"), primary_key=True)
    game_pk: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{_SCHEMA}.games.game_pk"), primary_key=True
    )
    season: Mapped[int] = mapped_column(SmallInteger)
    opponent_id: Mapped[int] = mapped_column(ForeignKey(f"{_SCHEMA}.teams.team_id"))
    is_home: Mapped[bool] = mapped_column(Boolean)
    runs_for: Mapped[int] = mapped_column(SmallInteger, default=0)
    runs_against: Mapped[int] = mapped_column(SmallInteger, default=0)
    hits_for: Mapped[int] = mapped_column(SmallInteger, default=0)
    hits_against: Mapped[int] = mapped_column(SmallInteger, default=0)
    won: Mapped[bool | None] = mapped_column(Boolean)


class PlayerSeasonStat(Base):
    __tablename__ = "player_season_stats"
    __table_args__ = {"schema": _SCHEMA}

    player_id: Mapped[int] = mapped_column(
        ForeignKey(f"{_SCHEMA}.players.player_id"), primary_key=True
    )
    season: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    group: Mapped[str] = mapped_column(String(8), primary_key=True)  # batting | pitching
    split: Mapped[str] = mapped_column(String(12), primary_key=True)  # all vs_L vs_R home away
    through_date: Mapped[str | None] = mapped_column(String(10))  # ISO date; NULL = full season
    games: Mapped[int] = mapped_column(SmallInteger, default=0)
    stat_json: Mapped[dict[str, Any]] = mapped_column(JSONB)


class TeamSeasonStat(Base):
    __tablename__ = "team_season_stats"
    __table_args__ = {"schema": _SCHEMA}

    team_id: Mapped[int] = mapped_column(ForeignKey(f"{_SCHEMA}.teams.team_id"), primary_key=True)
    season: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    split: Mapped[str] = mapped_column(String(12), primary_key=True)
    through_date: Mapped[str | None] = mapped_column(String(10))
    stat_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
