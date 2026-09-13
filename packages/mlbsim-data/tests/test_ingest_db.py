"""End-to-end ingest against a live database (schema must be migrated).

Runs only under ``-m integration``. Sources are still faked from fixtures so the
test is deterministic and offline; everything below the parser is real.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from mlbsim_core import session_scope
from mlbsim_data import ingest
from mlbsim_data.models import BattingGameLog, Game, PlateAppearance
from mlbsim_data.query import describe_game

pytestmark = pytest.mark.integration

_WAREHOUSE_TABLES = [
    "plate_appearances",
    "pitches",
    "batted_balls",
    "batting_game_logs",
    "pitching_game_logs",
    "team_game_logs",
    "lineups",
    "game_probables",
    "game_weather",
    "games",
    "players",
    "teams",
    "parks",
]


@pytest.fixture
def clean_warehouse():
    with session_scope() as s:
        s.execute(
            text("TRUNCATE " + ", ".join(f"warehouse.{t}" for t in _WAREHOUSE_TABLES) + " CASCADE")
        )
    yield


@pytest.fixture(autouse=True)
def _fake_sources(monkeypatch, game_feed, game_boxscore):
    monkeypatch.setattr(ingest.mlb_statsapi, "game_feed", lambda pk, **kw: game_feed)
    monkeypatch.setattr(ingest.mlb_statsapi, "boxscore", lambda pk, **kw: game_boxscore)


def test_ingest_then_query(clean_warehouse):
    summary = ingest.ingest_game(744914, assume_final=True)
    assert summary.plate_appearances == 33

    with session_scope() as s:
        assert s.get(Game, 744914) is not None
        assert s.query(PlateAppearance).filter_by(game_pk=744914).count() == 33
        assert s.query(BattingGameLog).filter_by(game_pk=744914).count() == 19

    described = describe_game(744914)
    assert "Blue Jays" in described["matchup"]
    assert described["plate_appearances"] == 33
    assert described["event_breakdown"]["home_run"] == 1
    assert described["pitches"] > 100


def test_ingest_is_idempotent(clean_warehouse):
    ingest.ingest_game(744914, assume_final=True)
    ingest.ingest_game(744914, assume_final=True)  # no IntegrityError, no duplicates
    with session_scope() as s:
        assert s.query(PlateAppearance).filter_by(game_pk=744914).count() == 33
