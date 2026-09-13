"""Season-stat rollups against a live database (schema must be migrated)."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from mlbsim_core import session_scope
from mlbsim_data import ingest
from mlbsim_data.models import PlayerSeasonStat, TeamSeasonStat
from mlbsim_data.transform import rebuild_season_stats, rebuild_team_season_stats

pytestmark = pytest.mark.integration

_TABLES = [
    "plate_appearances",
    "pitches",
    "batted_balls",
    "batting_game_logs",
    "pitching_game_logs",
    "team_game_logs",
    "player_season_stats",
    "team_season_stats",
    "lineups",
    "game_probables",
    "game_weather",
    "games",
    "players",
    "teams",
    "parks",
]


@pytest.fixture
def ingested_game(monkeypatch, game_feed, game_boxscore):
    monkeypatch.setattr(ingest.mlb_statsapi, "game_feed", lambda pk, **kw: game_feed)
    monkeypatch.setattr(ingest.mlb_statsapi, "boxscore", lambda pk, **kw: game_boxscore)
    with session_scope() as s:
        s.execute(text("TRUNCATE " + ", ".join(f"warehouse.{t}" for t in _TABLES) + " CASCADE"))
    ingest.ingest_game(744914, assume_final=True)
    yield


def test_rebuild_player_and_team_season_stats(ingested_game):
    n_players = rebuild_season_stats(2024)
    n_teams = rebuild_team_season_stats(2024)
    assert n_players > 20
    assert n_teams == 2

    with session_scope() as s:
        bichette = s.get(PlayerSeasonStat, (666182, 2024, "batting", "all"))
        assert bichette is not None
        assert bichette.stat_json["ab"] == 3
        assert bichette.stat_json["bb"] == 1

        # vs_R / vs_L splits come from the play-by-play
        splits = {
            r.split
            for r in s.query(PlayerSeasonStat).filter_by(group="batting").all()
            if r.split != "all"
        }
        assert splits <= {"vs_L", "vs_R"}
        assert splits  # at least one handedness split produced

        tor = s.get(TeamSeasonStat, (141, 2024, "all"))
        assert tor.stat_json["rf"] == 1
        assert tor.stat_json["ra"] == 3
        assert tor.stat_json["l"] == 1


def test_through_date_filters_out_later_games(ingested_game):
    # game is 2024-07-01; a cutoff before it yields no player rows
    assert rebuild_season_stats(2024, through_date="2024-06-30") == 0
    assert rebuild_season_stats(2024, through_date="2024-07-01") > 20
