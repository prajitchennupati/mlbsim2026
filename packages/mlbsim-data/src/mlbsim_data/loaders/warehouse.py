"""Entity-specific upserts into the ``warehouse`` schema.

Each helper wraps :func:`mlbsim_data.loaders.upsert.upsert` with the right
conflict key and, where a table is co-owned, a restricted ``update_columns`` set
so one source does not clobber columns another source is authoritative for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy.orm import Session

from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import (
    BattedBall,
    BattingGameLog,
    Game,
    GameProbable,
    GameWeather,
    Lineup,
    Park,
    Pitch,
    PitchingGameLog,
    PlateAppearance,
    Player,
    Team,
    TeamGameLog,
)

Rows = Sequence[dict[str, Any]]

# Columns the Chadwick crosswalk owns on `players`; never overwritten from game feeds.
_PLAYER_FROM_FEED = ("full_name", "bats", "throws", "birth_date", "primary_pos", "debut_date")


def load_parks(s: Session, rows: Rows) -> int:
    return upsert(s, Park, rows, index_elements=["park_id"])


def load_teams(s: Session, rows: Rows) -> int:
    return upsert(s, Team, rows, index_elements=["team_id"])


def load_players(s: Session, rows: Rows) -> int:
    return upsert(s, Player, rows, index_elements=["player_id"], update_columns=_PLAYER_FROM_FEED)


def load_games(s: Session, rows: Rows) -> int:
    return upsert(s, Game, rows, index_elements=["game_pk"])


def load_game_weather(s: Session, rows: Rows) -> int:
    return upsert(s, GameWeather, rows, index_elements=["game_pk"])


def load_game_probables(s: Session, rows: Rows) -> int:
    return upsert(s, GameProbable, rows, index_elements=["game_pk", "team_id"])


def load_lineups(s: Session, rows: Rows) -> int:
    return upsert(
        s, Lineup, rows, index_elements=["game_pk", "team_id", "batting_order", "slot_sequence"]
    )


def load_plate_appearances(s: Session, rows: Rows) -> int:
    return upsert(s, PlateAppearance, rows, index_elements=["pa_id"])


def load_pitches(s: Session, rows: Rows) -> int:
    return upsert(s, Pitch, rows, index_elements=["pitch_id"])


def load_batted_balls(s: Session, rows: Rows) -> int:
    return upsert(s, BattedBall, rows, index_elements=["pa_id"])


def load_batting_logs(s: Session, rows: Rows) -> int:
    return upsert(s, BattingGameLog, rows, index_elements=["player_id", "game_pk"])


def load_pitching_logs(s: Session, rows: Rows) -> int:
    return upsert(s, PitchingGameLog, rows, index_elements=["player_id", "game_pk"])


def load_team_logs(s: Session, rows: Rows) -> int:
    return upsert(s, TeamGameLog, rows, index_elements=["team_id", "game_pk"])
