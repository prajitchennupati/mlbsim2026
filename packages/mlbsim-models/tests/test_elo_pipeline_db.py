"""Elo build + evaluate against a live database (schema must be migrated)."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select, text

from mlbsim_core import session_scope
from mlbsim_data.models import (
    CalibrationBin,
    EloRating,
    Game,
    GamePrediction,
    ModelEvalRun,
    ModelVersion,
)
from mlbsim_models.pipelines import build_elo, evaluate_model, predict_elo_date

pytestmark = pytest.mark.integration

_TEAMS = [(1, "AAA"), (2, "BBB"), (3, "CCC"), (4, "DDD")]
_STRENGTH = {1: 0.68, 2: 0.55, 3: 0.45, 4: 0.32}  # true home-team-agnostic win rates


@pytest.fixture
def synthetic_games():
    with session_scope() as s:
        # elo_ratings has no FK to teams, so CASCADE won't reach it — clear it explicitly.
        s.execute(
            text(
                "TRUNCATE warehouse.games, warehouse.teams, warehouse.elo_ratings, "
                "serving.model_versions CASCADE"
            )
        )
        s.execute(
            text(
                "INSERT INTO warehouse.teams (team_id, abbr, name) VALUES "
                + ",".join(f"({tid},'{ab}','{ab}')" for tid, ab in _TEAMS)
            )
        )
        rows = []
        pk = 900000
        rng = __import__("random").Random(7)
        for season in (2023, 2024):
            day = dt.date(season, 4, 1)
            for _ in range(120):
                h, a = rng.sample([1, 2, 3, 4], 2)
                # crude Bradley-Terry-ish outcome from true strengths
                ph = _STRENGTH[h] / (_STRENGTH[h] + _STRENGTH[a])
                hs, as_ = (5, 2) if rng.random() < ph else (2, 5)
                rows.append(
                    {
                        "game_pk": pk,
                        "season": season,
                        "game_date": day,
                        "game_type": "R",
                        "status": "Final",
                        "home_team_id": h,
                        "away_team_id": a,
                        "home_score": hs,
                        "away_score": as_,
                        "dh_game_num": 1,
                        "is_doubleheader": False,
                        "scheduled_innings": 9,
                    }
                )
                pk += 1
                day += dt.timedelta(days=1)
        s.execute(Game.__table__.insert(), rows)
    yield


def test_build_elo_populates_ratings_and_predictions(synthetic_games):
    summary = build_elo()
    assert summary.games_processed == 240
    assert summary.prediction_rows == 240

    with session_scope() as s:
        assert s.get(ModelVersion, "elo_v1") is not None
        assert s.scalar(select(func.count()).select_from(EloRating)) > 0
        assert (
            s.scalar(
                select(func.count())
                .select_from(GamePrediction)
                .where(GamePrediction.model_id == "elo_v1")
            )
            == 240
        )
        # strongest team ends with the highest latest rating
        latest = {}
        for r in s.execute(
            select(EloRating.entity_id, EloRating.rating, EloRating.as_of_date).order_by(
                EloRating.as_of_date
            )
        ):
            latest[r.entity_id] = float(r.rating)
        assert latest[1] == max(latest.values())
        assert latest[4] == min(latest.values())


def test_build_elo_is_rebuildable(synthetic_games):
    build_elo()
    build_elo()  # rebuild=True wipes prior elo_v1 predictions first
    with session_scope() as s:
        assert (
            s.scalar(
                select(func.count())
                .select_from(GamePrediction)
                .where(GamePrediction.model_id == "elo_v1")
            )
            == 240
        )


def test_evaluate_model_beats_coin_flip(synthetic_games):
    build_elo()
    m = evaluate_model("elo_v1", season=2024)
    assert m["model"]["n"] > 100
    assert 0.0 < m["model"]["log_loss"] < 0.69  # better than coin flip (ln 2 ≈ 0.693)
    assert m["beats_coin_flip_logloss"] is True

    with session_scope() as s:
        ev = s.scalar(select(ModelEvalRun).where(ModelEvalRun.split_name == "season_2024"))
        assert ev is not None
        assert (
            s.scalar(
                select(func.count())
                .select_from(CalibrationBin)
                .where(CalibrationBin.eval_id == ev.eval_id)
            )
            > 0
        )


def test_evaluate_model_without_predictions_raises(synthetic_games):
    with pytest.raises(LookupError):
        evaluate_model("does_not_exist", season=2024)


def test_predict_elo_date_scores_unplayed_games(synthetic_games):
    build_elo()  # establishes ratings from the synthetic 2023/2024 games

    upcoming = dt.date(2025, 4, 1)
    with session_scope() as s:
        s.execute(
            Game.__table__.insert(),
            [
                {
                    "game_pk": 999001,
                    "season": 2025,
                    "game_date": upcoming,
                    "game_type": "R",
                    "status": "Scheduled",
                    "home_team_id": 1,  # strongest synthetic team
                    "away_team_id": 4,  # weakest synthetic team
                    "home_score": None,
                    "away_score": None,
                    "dh_game_num": 1,
                    "is_doubleheader": False,
                    "scheduled_innings": 9,
                }
            ],
        )

    summary = predict_elo_date(upcoming.isoformat())
    assert summary.games_processed == 1
    assert summary.prediction_rows == 1

    with session_scope() as s:
        pred = s.scalar(
            select(GamePrediction).where(
                GamePrediction.game_pk == 999001, GamePrediction.model_id == "elo_v1"
            )
        )
        assert pred is not None
        assert float(pred.home_win_prob) > 0.5  # team 1 rated above team 4

    # idempotent — replaces rather than duplicating the game's elo_v1 row
    predict_elo_date(upcoming.isoformat())
    with session_scope() as s:
        n = s.scalar(
            select(func.count())
            .select_from(GamePrediction)
            .where(GamePrediction.game_pk == 999001, GamePrediction.model_id == "elo_v1")
        )
        assert n == 1


def test_predict_elo_date_skips_already_finished_games(synthetic_games):
    build_elo()
    # every synthetic game already carries a final score -> nothing left to predict
    summary = predict_elo_date("2023-04-01", "2024-12-31")
    assert summary.games_processed == 0
    assert summary.prediction_rows == 0
