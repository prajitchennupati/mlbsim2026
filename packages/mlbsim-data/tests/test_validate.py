from __future__ import annotations

import pytest
from pandera.errors import SchemaErrors

from mlbsim_data.validate import GAME_SCHEMA, PLATE_APPEARANCE_SCHEMA, validate_rows


def test_valid_game_rows_pass():
    rows = [
        {
            "game_pk": 1,
            "season": 2024,
            "game_date": "2024-07-01",
            "game_type": "R",
            "home_team_id": 141,
            "away_team_id": 117,
            "home_score": 1,
            "away_score": 3,
        }
    ]
    assert validate_rows(rows, GAME_SCHEMA) == rows


def test_bad_game_type_rejected():
    rows = [
        {
            "game_pk": 1,
            "season": 2024,
            "game_date": "2024-07-01",
            "game_type": "ZZ",
            "home_team_id": 1,
            "away_team_id": 2,
            "home_score": None,
            "away_score": None,
        }
    ]
    with pytest.raises(SchemaErrors):
        validate_rows(rows, GAME_SCHEMA)


def test_outs_end_before_start_rejected():
    rows = [
        {
            "pa_id": 10,
            "game_pk": 1,
            "inning": 3,
            "half": "top",
            "batter_id": 5,
            "pitcher_id": 6,
            "outs_start": 2,
            "outs_end": 1,
            "base_state_start": 0,
            "base_state_end": 0,
            "event_type": "strikeout",
            "rbi": 0,
            "runs_on_play": 0,
        }
    ]
    with pytest.raises(SchemaErrors):
        validate_rows(rows, PLATE_APPEARANCE_SCHEMA)


def test_empty_rows_short_circuit():
    assert validate_rows([], GAME_SCHEMA) == []
