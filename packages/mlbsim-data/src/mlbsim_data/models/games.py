"""Game-level fact tables (``warehouse`` schema)."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import BigInteger, Boolean, ForeignKey, Index, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from mlbsim_core.db import Base
from mlbsim_data.models._types import ingested_ts, ts, ts_opt

_SCHEMA = "warehouse"

# ``Game.status`` values (the Stats API's ``status.detailedState``) that mean the
# game actually has a final result. A non-null score is *not* a reliable signal
# on its own — the schedule endpoint reports 0-0 for plenty of games that
# haven't started yet, not a missing/null score.
FINAL_STATUSES: frozenset[str] = frozenset({"Final", "Game Over", "Completed Early"})

# Statuses meaning no game will be played at this game_pk at all — neither a
# result to grade nor a pre-game matchup worth predicting.
VOID_STATUSES: frozenset[str] = frozenset({"Postponed", "Cancelled", "Suspended"})


def is_game_final(status: str | None) -> bool:
    """Whether ``status`` means the game has a real final result."""
    return status in FINAL_STATUSES


class Game(Base):
    __tablename__ = "games"
    __table_args__ = (
        Index("ix_games_date", "game_date"),
        Index("ix_games_season", "season"),
        {"schema": _SCHEMA},
    )

    game_pk: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    season: Mapped[int] = mapped_column(SmallInteger)
    game_date: Mapped[dt.date]
    game_type: Mapped[str] = mapped_column(String(2))  # R F D L W S A E
    status: Mapped[str] = mapped_column(String(24))  # detailedState
    home_team_id: Mapped[int] = mapped_column(ForeignKey(f"{_SCHEMA}.teams.team_id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey(f"{_SCHEMA}.teams.team_id"))
    park_id: Mapped[int | None] = mapped_column(ForeignKey(f"{_SCHEMA}.parks.park_id"))
    scheduled_start_utc: Mapped[ts_opt]
    day_night: Mapped[str | None] = mapped_column(String(5))
    is_doubleheader: Mapped[bool] = mapped_column(Boolean, default=False)
    dh_game_num: Mapped[int] = mapped_column(SmallInteger, default=1)
    scheduled_innings: Mapped[int] = mapped_column(SmallInteger, default=9)

    home_score: Mapped[int | None] = mapped_column(SmallInteger)
    away_score: Mapped[int | None] = mapped_column(SmallInteger)
    n_innings: Mapped[int | None] = mapped_column(SmallInteger)

    home_sp_id: Mapped[int | None] = mapped_column(ForeignKey(f"{_SCHEMA}.players.player_id"))
    away_sp_id: Mapped[int | None] = mapped_column(ForeignKey(f"{_SCHEMA}.players.player_id"))
    winning_pitcher_id: Mapped[int | None] = mapped_column(
        ForeignKey(f"{_SCHEMA}.players.player_id")
    )
    losing_pitcher_id: Mapped[int | None] = mapped_column(
        ForeignKey(f"{_SCHEMA}.players.player_id")
    )
    save_pitcher_id: Mapped[int | None] = mapped_column(ForeignKey(f"{_SCHEMA}.players.player_id"))

    ingested_at: Mapped[ingested_ts]


class GameWeather(Base):
    __tablename__ = "game_weather"
    __table_args__ = {"schema": _SCHEMA}

    game_pk: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{_SCHEMA}.games.game_pk"), primary_key=True
    )
    forecast_ts: Mapped[ts_opt]
    source: Mapped[str] = mapped_column(String(16), default="statsapi")  # statsapi | open-meteo
    temp_f: Mapped[int | None] = mapped_column(SmallInteger)
    wind_mph: Mapped[int | None] = mapped_column(SmallInteger)
    wind_dir: Mapped[str | None] = mapped_column(String(24))
    condition: Mapped[str | None] = mapped_column(String(40))
    humidity: Mapped[int | None] = mapped_column(SmallInteger)
    observed_temp_f: Mapped[int | None] = mapped_column(SmallInteger)


class GameProbable(Base):
    __tablename__ = "game_probables"
    __table_args__ = {"schema": _SCHEMA}

    game_pk: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{_SCHEMA}.games.game_pk"), primary_key=True
    )
    team_id: Mapped[int] = mapped_column(ForeignKey(f"{_SCHEMA}.teams.team_id"), primary_key=True)
    probable_pitcher_id: Mapped[int | None] = mapped_column(
        ForeignKey(f"{_SCHEMA}.players.player_id")
    )
    source_ts: Mapped[ts]


class Lineup(Base):
    __tablename__ = "lineups"
    __table_args__ = {"schema": _SCHEMA}

    game_pk: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{_SCHEMA}.games.game_pk"), primary_key=True
    )
    team_id: Mapped[int] = mapped_column(ForeignKey(f"{_SCHEMA}.teams.team_id"), primary_key=True)
    batting_order: Mapped[int] = mapped_column(SmallInteger, primary_key=True)  # 1..9
    slot_sequence: Mapped[int] = mapped_column(SmallInteger, primary_key=True, default=0)
    player_id: Mapped[int] = mapped_column(ForeignKey(f"{_SCHEMA}.players.player_id"))
    position: Mapped[str | None] = mapped_column(String(3))
    source: Mapped[str] = mapped_column(String(10))  # projected | actual
    source_ts: Mapped[ts]
