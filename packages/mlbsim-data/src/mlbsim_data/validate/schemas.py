"""Pandera schemas for the highest-risk parsed row sets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

_GAME_TYPES = ["R", "F", "D", "L", "W", "S", "A", "E", "I", "P"]
_EVENT_TYPES = [
    "single",
    "double",
    "triple",
    "home_run",
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "strikeout",
    "strikeout_double_play",
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "double_play",
    "triple_play",
    "sac_fly",
    "sac_bunt",
    "sac_fly_double_play",
    "sac_bunt_double_play",
    "field_error",
    "fielders_choice",
    "fielders_choice_out",
    "catcher_interf",
    "batter_interference",
    "fan_interference",
    "other_out",
    "unknown",
]

GAME_SCHEMA = DataFrameSchema(
    {
        "game_pk": Column(int, unique=True),
        "season": Column(int, Check.in_range(1900, 2100)),
        "game_date": Column("datetime64[ns]", nullable=True),
        "game_type": Column(str, Check.isin(_GAME_TYPES)),
        "home_team_id": Column(int),
        "away_team_id": Column(int),
        "home_score": Column("Int64", Check.in_range(0, 60), nullable=True),
        "away_score": Column("Int64", Check.in_range(0, 60), nullable=True),
    },
    strict=False,
    coerce=True,
)

PLATE_APPEARANCE_SCHEMA = DataFrameSchema(
    {
        "pa_id": Column(int, unique=True),
        "game_pk": Column(int),
        "inning": Column(int, Check.in_range(1, 30)),
        "half": Column(str, Check.isin(["top", "bottom"])),
        "batter_id": Column(int),
        "pitcher_id": Column(int),
        "outs_start": Column(int, Check.in_range(0, 3)),
        "outs_end": Column(int, Check.in_range(0, 3)),
        "base_state_start": Column(int, Check.in_range(0, 7)),
        "base_state_end": Column(int, Check.in_range(0, 7)),
        "event_type": Column(str, Check.isin(_EVENT_TYPES)),
        "rbi": Column(int, Check.in_range(0, 4)),
        "runs_on_play": Column(int, Check.in_range(0, 4)),
    },
    strict=False,
    coerce=True,
    checks=Check(
        lambda df: df["outs_end"] >= df["outs_start"],
        error="outs_end must be >= outs_start",
    ),
)

BATTING_GAME_LOG_SCHEMA = DataFrameSchema(
    {
        "player_id": Column(int),
        "game_pk": Column(int),
        "team_id": Column(int),
        "pa": Column(int, Check.in_range(0, 15)),
        "ab": Column(int, Check.in_range(0, 15)),
        "h": Column(int, Check.in_range(0, 10)),
        "hr": Column(int, Check.in_range(0, 6)),
        "tb": Column(int, Check.in_range(0, 20)),
    },
    strict=False,
    coerce=True,
    checks=Check(lambda df: df["ab"] <= df["pa"], error="ab must be <= pa"),
)


def validate_rows(rows: Sequence[dict[str, Any]], schema: DataFrameSchema) -> list[dict[str, Any]]:
    """Validate ``rows`` against ``schema``; return them unchanged on success."""
    materialised = list(rows)
    if not materialised:
        return materialised
    schema.validate(pd.DataFrame(materialised), lazy=True)
    return materialised
