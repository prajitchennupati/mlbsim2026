"""Player-prediction + Marcel pipelines against a live database."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select, text

from mlbsim_core import session_scope
from mlbsim_data.models import (
    Game,
    Lineup,
    PlayerPrediction,
    PlayerSeasonStat,
    SimulationRun,
)
from mlbsim_models.pipelines import (
    marcel_for_player,
    predict_players_for_game,
    store_marcel_projections,
)

pytestmark = pytest.mark.integration

_GAME_PK = 820001
_HOME, _AWAY = 30, 40
_HOME_SP, _AWAY_SP = 6001, 6002
_HOME_BATS = [7000 + i for i in range(9)]
_AWAY_BATS = [7100 + i for i in range(9)]
_TABLES = [
    "player_predictions",
    "game_sim_results",
    "simulation_runs",
    "lineups",
    "player_season_stats",
    "games",
    "players",
    "teams",
]


def _bat(hr: int, so: int) -> dict:
    return {
        "pa": 620,
        "ab": 560,
        "h": 155,
        "d2b": 30,
        "t3b": 2,
        "hr": hr,
        "bb": 55,
        "hbp": 5,
        "so": so,
    }


@pytest.fixture
def game_ready():
    with session_scope() as s:
        s.execute(
            text(
                "TRUNCATE "
                + ", ".join(
                    f"warehouse.{t}"
                    for t in _TABLES
                    if t not in ("player_predictions", "simulation_runs", "game_sim_results")
                )
                + " CASCADE"
            )
        )
        s.execute(text("TRUNCATE serving.simulation_runs, serving.model_versions CASCADE"))
        s.execute(
            text(
                "INSERT INTO warehouse.teams (team_id, abbr, name) VALUES "
                f"({_HOME},'HOM','HOM'),({_AWAY},'AWY','AWY')"
            )
        )
        pids = [*_HOME_BATS, *_AWAY_BATS, _HOME_SP, _AWAY_SP]
        s.execute(
            text(
                "INSERT INTO warehouse.players (player_id, full_name, birth_date) VALUES "
                + ",".join(f"({p},'P{p}','1996-05-01')" for p in pids)
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
                    "home_score": 4,
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
                for team, bats in ((_HOME, _HOME_BATS), (_AWAY, _AWAY_BATS))
                for i, pid in enumerate(bats)
            ],
        )
        ss = []
        for pid in _HOME_BATS:
            ss.append(
                {
                    "player_id": pid,
                    "season": 2024,
                    "group": "batting",
                    "split": "all",
                    "through_date": "2024-06-30",
                    "games": 80,
                    "stat_json": _bat(34, 110),
                }
            )
        for pid in _AWAY_BATS:
            ss.append(
                {
                    "player_id": pid,
                    "season": 2024,
                    "group": "batting",
                    "split": "all",
                    "through_date": "2024-06-30",
                    "games": 80,
                    "stat_json": _bat(12, 165),
                }
            )
        # prior seasons for Marcel (2023, 2022)
        for yr in (2023, 2022):
            for pid in _HOME_BATS:
                ss.append(
                    {
                        "player_id": pid,
                        "season": yr,
                        "group": "batting",
                        "split": "all",
                        "through_date": None,
                        "games": 150,
                        "stat_json": _bat(30, 120),
                    }
                )
        for pid in (_HOME_SP, _AWAY_SP):
            ss.append(
                {
                    "player_id": pid,
                    "season": 2024,
                    "group": "pitching",
                    "split": "all",
                    "through_date": "2024-06-30",
                    "games": 15,
                    "stat_json": {"bf": 400, "so": 105, "bb": 28, "hbp": 4, "hr": 11, "h": 88},
                }
            )
        s.execute(PlayerSeasonStat.__table__.insert(), ss)
    yield


def test_predict_players_writes_batter_and_pitcher_rows(game_ready):
    summary = predict_players_for_game(_GAME_PK, n_sims=3000, seed=1)
    assert summary.batters == 18
    assert summary.pitchers == 2
    assert summary.sim_run_id is not None

    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(SimulationRun)) == 1
        bats = list(s.scalars(select(PlayerPrediction).where(PlayerPrediction.role == "bat")))
        assert len(bats) == 18
        # strong home hitter has a higher P(HR) than the weak away hitter
        home_hr = next(b.prob_json["p_hr"] for b in bats if b.player_id == _HOME_BATS[3])
        away_hr = next(b.prob_json["p_hr"] for b in bats if b.player_id == _AWAY_BATS[3])
        assert home_hr > away_hr
        for b in bats:
            assert 0.0 <= b.prob_json["p_1plus_h"] <= 1.0
            assert set(b.proj_json) >= {"pa", "h", "hr", "k"}
        sp = s.scalar(select(PlayerPrediction).where(PlayerPrediction.role == "pitch").limit(1))
        assert "p_5plus_k" in sp.prob_json and "mean_ip" in sp.proj_json


def test_predict_players_is_rebuildable(game_ready):
    predict_players_for_game(_GAME_PK, n_sims=1500, seed=1)
    predict_players_for_game(_GAME_PK, n_sims=1500, seed=2)
    with session_scope() as s:
        assert (
            s.scalar(
                select(func.count())
                .select_from(PlayerPrediction)
                .where(PlayerPrediction.model_id == "sim_v1")
            )
            == 20
        )


def test_marcel_from_db_and_store(game_ready):
    proj = marcel_for_player(_HOME_BATS[0], 2025)
    assert proj.proj_pa > 200
    assert 0.99 < proj.rates.sum() < 1.01

    n = store_marcel_projections(2025)
    assert n > 0
    with session_scope() as s:
        row = s.get(PlayerSeasonStat, (_HOME_BATS[0], 2025, "batting", "marcel"))
        assert row is not None
        assert "rates" in row.stat_json and len(row.stat_json["rates"]) == 8
