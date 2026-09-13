"""simulate_game_from_db against a live database (schema must be migrated)."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select, text

from mlbsim_core import session_scope
from mlbsim_data.models import (
    Game,
    GameSimResultRow,
    Lineup,
    PlayerSeasonStat,
    SimulationRun,
)
from mlbsim_engine.pipeline import simulate_game_from_db

pytestmark = pytest.mark.integration

_GAME_PK = 810001
_HOME, _AWAY = 10, 20
_HOME_SP, _AWAY_SP = 5001, 5002
_TABLES = [
    "lineups",
    "player_season_stats",
    "games",
    "players",
    "teams",
]


def _bat_line(power: bool) -> dict:
    return {
        "pa": 600,
        "ab": 540,
        "h": 175 if power else 135,
        "d2b": 32,
        "t3b": 2,
        "hr": 35 if power else 12,
        "bb": 55,
        "hbp": 5,
        "so": 120 if power else 150,
    }


@pytest.fixture
def one_game():
    home_bats = [3000 + i for i in range(9)]
    away_bats = [4000 + i for i in range(9)]
    with session_scope() as s:
        s.execute(text("TRUNCATE " + ", ".join(f"warehouse.{t}" for t in _TABLES) + " CASCADE"))
        s.execute(text("TRUNCATE serving.simulation_runs, serving.game_sim_results CASCADE"))
        s.execute(
            text(
                "INSERT INTO warehouse.teams (team_id, abbr, name) VALUES "
                f"({_HOME},'HHH','HHH'),({_AWAY},'AAA','AAA')"
            )
        )
        pids = [*home_bats, *away_bats, _HOME_SP, _AWAY_SP]
        s.execute(
            text(
                "INSERT INTO warehouse.players (player_id, full_name) VALUES "
                + ",".join(f"({p},'P{p}')" for p in pids)
            )
        )
        s.execute(
            Game.__table__.insert(),
            [
                {
                    "game_pk": _GAME_PK,
                    "season": 2024,
                    "game_date": dt.date(2024, 7, 1),
                    "game_type": "R",
                    "status": "Final",
                    "home_team_id": _HOME,
                    "away_team_id": _AWAY,
                    "home_score": 5,
                    "away_score": 3,
                    "home_sp_id": _HOME_SP,
                    "away_sp_id": _AWAY_SP,
                    "dh_game_num": 1,
                    "is_doubleheader": False,
                    "scheduled_innings": 9,
                }
            ],
        )
        s.execute(
            Lineup.__table__.insert(),
            [
                {
                    "game_pk": _GAME_PK,
                    "team_id": team,
                    "batting_order": i + 1,
                    "slot_sequence": 0,
                    "player_id": pid,
                    "position": "DH",
                    "source": "actual",
                    "source_ts": dt.datetime(2024, 7, 1, tzinfo=dt.UTC),
                }
                for team, bats in ((_HOME, home_bats), (_AWAY, away_bats))
                for i, pid in enumerate(bats)
            ],
        )
        rows = []
        for pid in home_bats:  # strong home offense
            rows.append(
                {
                    "player_id": pid,
                    "season": 2024,
                    "group": "batting",
                    "split": "all",
                    "through_date": "2024-06-30",
                    "games": 80,
                    "stat_json": _bat_line(True),
                }
            )
        for pid in away_bats:  # weak away offense
            rows.append(
                {
                    "player_id": pid,
                    "season": 2024,
                    "group": "batting",
                    "split": "all",
                    "through_date": "2024-06-30",
                    "games": 80,
                    "stat_json": _bat_line(False),
                }
            )
        for pid in (_HOME_SP, _AWAY_SP):
            rows.append(
                {
                    "player_id": pid,
                    "season": 2024,
                    "group": "pitching",
                    "split": "all",
                    "through_date": "2024-06-30",
                    "games": 15,
                    "stat_json": {"bf": 400, "so": 100, "bb": 30, "hbp": 4, "hr": 12, "h": 90},
                }
            )
        s.execute(PlayerSeasonStat.__table__.insert(), rows)
    yield


def test_simulate_from_db_persists_and_summarises(one_game):
    summary = simulate_game_from_db(_GAME_PK, n_sims=3000, seed=1)
    assert summary["n_sims"] == 3000
    assert 0.0 < summary["home_win_prob"] < 1.0
    # strong home offense vs weak away -> home favoured and outscores
    assert summary["home_win_prob"] > 0.6
    assert summary["exp_home_runs"] > summary["exp_away_runs"]
    assert "sim_run_id" in summary
    assert len(summary["home_batting"]) == 9

    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(SimulationRun)) == 1
        row = s.scalar(select(GameSimResultRow).where(GameSimResultRow.game_pk == _GAME_PK))
        assert row is not None
        assert float(row.home_win_prob) == pytest.approx(summary["home_win_prob"], abs=1e-4)
        assert row.summary_json["most_likely_scores"]


def test_missing_game_raises(one_game):
    with pytest.raises(LookupError):
        simulate_game_from_db(999999, n_sims=100)
