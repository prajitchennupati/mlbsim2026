"""End-to-end endpoint tests against a live, migrated database."""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from mlbsim_api.main import create_app
from mlbsim_core import session_scope
from mlbsim_data.models import (
    CalibrationBin,
    Game,
    GamePrediction,
    GameSimResultRow,
    ModelEvalRun,
    ModelVersion,
    PlayerPrediction,
    PlayerSeasonStat,
    PredictionOutcome,
    SeasonSimTeamResult,
    SeriesPrediction,
    SimulationRun,
    TeamGameLog,
)

pytestmark = pytest.mark.integration

client = TestClient(create_app())

_TODAY = dt.date.today()
_GAME_PK = 990001
_HOME, _AWAY = 111, 222
_WAREHOUSE = ["team_game_logs", "players", "games", "teams"]
_SERVING = [
    "player_predictions",
    "game_predictions",
    "prediction_outcomes",
    "model_eval_runs",
    "calibration_bins",
    "game_sim_results",
    "season_sim_team_results",
    "series_predictions",
    "simulation_runs",
    "model_versions",
]


@pytest.fixture
def seeded():
    now = dt.datetime.now(dt.UTC)
    with session_scope() as s:
        s.execute(text("TRUNCATE " + ", ".join(f"warehouse.{t}" for t in _WAREHOUSE) + " CASCADE"))
        s.execute(text("TRUNCATE " + ", ".join(f"serving.{t}" for t in _SERVING) + " CASCADE"))
        s.execute(
            text(
                "INSERT INTO warehouse.teams (team_id, abbr, name, league, division) VALUES "
                f"({_HOME},'HOM','Home','AL','E'),"
                f"({_AWAY},'AWY','Away','AL','E')"
            )
        )
        s.execute(
            text(
                "INSERT INTO warehouse.players (player_id, full_name) VALUES "
                "(5001,'SP H'),(5002,'SP A'),(6001,'Bat')"
            )
        )
        s.execute(
            Game.__table__.insert(),
            [
                {
                    "game_pk": _GAME_PK,
                    "season": _TODAY.year,
                    "game_date": _TODAY,
                    "game_type": "R",
                    "status": "Final",
                    "home_team_id": _HOME,
                    "away_team_id": _AWAY,
                    "home_score": 5,
                    "away_score": 3,
                    "home_sp_id": 5001,
                    "away_sp_id": 5002,
                    "dh_game_num": 1,
                    "is_doubleheader": False,
                    "scheduled_innings": 9,
                }
            ],
        )
        s.execute(
            TeamGameLog.__table__.insert(),
            [
                {
                    "team_id": _HOME,
                    "game_pk": _GAME_PK,
                    "season": _TODAY.year,
                    "opponent_id": _AWAY,
                    "is_home": True,
                    "runs_for": 5,
                    "runs_against": 3,
                    "hits_for": 8,
                    "hits_against": 6,
                    "won": True,
                },
            ],
        )
        s.execute(
            ModelVersion.__table__.insert(),
            [
                {
                    "model_id": "direct_v1",
                    "name": "Direct",
                    "kind": "logit",
                    "version": "1",
                    "trained_at": now,
                    "metrics_json": {"in_sample": {"win_auc": 0.6}},
                },
            ],
        )
        pred_id = s.execute(
            GamePrediction.__table__.insert().returning(GamePrediction.__table__.c.pred_id),
            [
                {
                    "game_pk": _GAME_PK,
                    "model_id": "direct_v1",
                    "created_at": now,
                    "is_live": False,
                    "home_win_prob": 0.58,
                    "away_win_prob": 0.42,
                    "exp_home_runs": 4.6,
                    "exp_away_runs": 4.0,
                    "p_extra_innings": 0.09,
                    "p_one_run_game": 0.3,
                    "home_score_dist": {"0": 0.05},
                    "total_runs_dist": {"8": 0.1},
                    "most_likely_scores": [{"home": 4, "away": 3, "p": 0.03}],
                    "factors": {
                        "top_factors": [
                            {
                                "feature": "d_win_pct",
                                "label": "season win rate",
                                "favours": "home",
                                "logit_contribution": 0.4,
                                "prob_shift": 0.08,
                            }
                        ]
                    },
                }
            ],
        ).scalar_one()
        s.execute(
            PredictionOutcome.__table__.insert(),
            [
                {
                    "pred_id": pred_id,
                    "game_pk": _GAME_PK,
                    "resolved_at": now,
                    "actual_home_score": 5,
                    "actual_away_score": 3,
                    "actual_winner": "home",
                    "correct": True,
                    "brier": 0.17,
                    "log_loss": 0.44,
                },
            ],
        )
        s.execute(
            PlayerPrediction.__table__.insert(),
            [
                {
                    "game_pk": _GAME_PK,
                    "player_id": 6001,
                    "model_id": "direct_v1",
                    "role": "bat",
                    "created_at": now,
                    "proj_json": {"pa": 4.2, "h": 1.1},
                    "prob_json": {"p_1plus_h": 0.7},
                },
            ],
        )
        s.execute(
            PlayerSeasonStat.__table__.insert(),
            [
                {
                    "player_id": 6001,
                    "season": 2025,
                    "group": "batting",
                    "split": "all",
                    "through_date": None,
                    "games": 150,
                    "stat_json": {"pa": 600, "hr": 25, "avg": 0.281},
                },
            ],
        )
        eval_id = s.execute(
            ModelEvalRun.__table__.insert().returning(ModelEvalRun.__table__.c.eval_id),
            [
                {
                    "model_id": "direct_v1",
                    "split_name": "season_2024",
                    "metrics_json": {"model": {"n": 100}},
                    "created_at": now,
                }
            ],
        ).scalar_one()
        s.execute(
            CalibrationBin.__table__.insert(),
            [
                {
                    "eval_id": eval_id,
                    "bin_lower": 0.5,
                    "bin_upper": 0.6,
                    "n": 40,
                    "mean_pred": 0.55,
                    "mean_actual": 0.53,
                },
            ],
        )
        run_id = s.execute(
            SimulationRun.__table__.insert().returning(SimulationRun.__table__.c.sim_run_id),
            [
                {
                    "scope": "game",
                    "target_id": _GAME_PK,
                    "engine_version": "0.1.0",
                    "n_sims": 5000,
                    "seed": 0,
                    "runtime_ms": 120,
                    "params_json": {},
                    "created_at": now,
                }
            ],
        ).scalar_one()
        s.execute(
            GameSimResultRow.__table__.insert(),
            [
                {
                    "sim_run_id": run_id,
                    "game_pk": _GAME_PK,
                    "home_win_prob": 0.57,
                    "exp_home_runs": 4.7,
                    "exp_away_runs": 4.1,
                    "summary_json": {
                        "inning_probs": {
                            "home": [
                                {
                                    "exp_runs": 0.5,
                                    "p_score_1plus": 0.26,
                                    "p_score_2plus": 0.13,
                                    "p_score_3plus": 0.06,
                                    "p_scoreless": 0.74,
                                }
                            ],
                            "away": [
                                {
                                    "exp_runs": 0.5,
                                    "p_score_1plus": 0.26,
                                    "p_score_2plus": 0.13,
                                    "p_score_3plus": 0.06,
                                    "p_scoreless": 0.74,
                                }
                            ],
                            "p_home_lead_after": [0.51],
                        }
                    },
                },
            ],
        )
        season_run = s.execute(
            SimulationRun.__table__.insert().returning(SimulationRun.__table__.c.sim_run_id),
            [
                {
                    "scope": "season",
                    "target_id": _TODAY.year,
                    "model_id": "direct_v1",
                    "engine_version": "0.1.0",
                    "n_sims": 10000,
                    "seed": 0,
                    "runtime_ms": 700,
                    "params_json": {"season": _TODAY.year, "as_of": _TODAY.isoformat()},
                    "created_at": now,
                }
            ],
        ).scalar_one()
        s.execute(
            SeasonSimTeamResult.__table__.insert(),
            [
                {
                    "sim_run_id": season_run,
                    "team_id": tid,
                    "p_playoffs": pp,
                    "p_division": pp / 2,
                    "p_wildcard": pp / 2,
                    "p_bye": pp / 3,
                    "p_pennant": pp / 4,
                    "p_world_series": pp / 8,
                    "exp_wins": 88.0,
                    "exp_losses": 74.0,
                    "exp_seed": 3.0,
                    "exp_postseason_game_wins": 2.0,
                    "win_dist": {"88": 100},
                }
                for tid, pp in ((_HOME, 0.8), (_AWAY, 0.2))
            ],
        )
        s.execute(
            SeriesPrediction.__table__.insert(),
            [
                {
                    "sim_run_id": season_run,
                    "round": "WS",
                    "best_of": 7,
                    "p_sweep": 0.12,
                    "exp_games": 5.8,
                    "game_count_dist": {"4": 0.12, "7": 0.31},
                },
            ],
        )
    yield pred_id


def test_games_list_and_detail(seeded):
    r = client.get(f"/api/v1/games?date={_TODAY.isoformat()}")
    assert r.status_code == 200
    body = r.json()
    assert body[0]["game_pk"] == _GAME_PK
    assert body[0]["home_win_prob"] == 0.58
    assert body[0]["home_abbr"] == "HOM"
    assert body[0]["is_final"] is True
    assert body[0]["predicted_winner"] == "home"
    assert body[0]["correct"] is True

    d = client.get(f"/api/v1/games/{_GAME_PK}").json()
    assert d["away_sp_id"] == 5002
    assert d["factors"]["top_factors"]
    assert d["is_final"] is True
    assert client.get("/api/v1/games/404040").status_code == 404


def test_game_subresources(seeded):
    assert client.get(f"/api/v1/games/{_GAME_PK}/simulation").json()["n_sims"] == 5000
    innings = client.get(f"/api/v1/games/{_GAME_PK}/innings").json()
    assert innings["source"] == "simulation"
    assert innings["rows"][0]["side"] == "home"
    players = client.get(f"/api/v1/games/{_GAME_PK}/players").json()
    assert players["batters"][0]["prob"]["p_1plus_h"] == 0.7


def test_teams_standings_playoffs(seeded):
    teams = client.get("/api/v1/teams").json()
    hom = next(t for t in teams if t["abbr"] == "HOM")
    assert hom["p_world_series"] == pytest.approx(0.1)
    assert client.get("/api/v1/teams/HOM").json()["wins"] == 1
    assert client.get("/api/v1/teams/HOM/schedule").json()[0]["game_pk"] == _GAME_PK
    assert client.get("/api/v1/teams/ZZZ").status_code == 404

    standings = client.get(f"/api/v1/standings?season={_TODAY.year}").json()
    assert standings[0]["wins"] == 1 and standings[0]["run_diff"] == 2

    po = client.get("/api/v1/playoffs").json()
    assert po["series"][0]["round"] == "WS"
    assert any(t["abbr"] == "HOM" for t in po["teams"])
    assert client.get("/api/v1/simulations/season/latest").json()["sim_run_id"] == po["sim_run_id"]


def test_models_and_predictions(seeded):
    models = client.get("/api/v1/models").json()
    assert any(m["model_id"] == "direct_v1" for m in models)
    ev = client.get("/api/v1/models/direct_v1/evaluation").json()
    assert ev["runs"][0]["split_name"] == "season_2024"
    assert ev["calibration"][0]["n"] == 40
    assert client.get("/api/v1/models/nope/evaluation").status_code == 404

    hist = client.get(f"/api/v1/predictions/history?date={_TODAY.isoformat()}").json()
    assert hist[0]["actual_winner"] == "home" and hist[0]["correct"] is True
    assert hist[0]["predicted_winner"] == "home"
    assert hist[0]["home_abbr"] == "HOM"
    graded = client.get("/api/v1/predictions/history?resolved=true").json()
    assert all(r["correct"] is not None for r in graded) and len(graded) >= 1

    summ = client.get("/api/v1/predictions/summary").json()
    assert summ["overall"]["n"] >= 1
    assert summ["overall"]["accuracy"] == pytest.approx(1.0)
    assert summ["pending"] == 0
    assert any(b["label"] == "direct_v1" for b in summ["by_model"])

    exp = client.get(f"/api/v1/predictions/{seeded}/explanation").json()
    assert exp["top_factors"][0]["label"] == "season win rate"
    assert "HOM" in exp["explanation"]
    assert client.get("/api/v1/predictions/8888888/explanation").status_code == 404


def test_player_card(seeded):
    r = client.get("/api/v1/players/6001")
    assert r.status_code == 200
    body = r.json()
    assert body["full_name"] == "Bat"
    assert body["season_lines"][0]["season"] == 2025
    assert body["season_lines"][0]["stats"]["hr"] == 25
    assert body["recent_predictions"][0]["game_pk"] == _GAME_PK
    assert body["recent_predictions"][0]["prob"]["p_1plus_h"] == 0.7

    assert client.get("/api/v1/players/424242").status_code == 404
