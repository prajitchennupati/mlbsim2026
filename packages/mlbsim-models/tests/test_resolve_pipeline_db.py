"""resolve_outcomes + rollup_eval against a live database (schema must be migrated)."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select, text

from mlbsim_core import session_scope
from mlbsim_data.models import (
    CalibrationBin,
    Game,
    GamePrediction,
    ModelEvalRun,
    ModelVersion,
    PredictionOutcome,
)
from mlbsim_models.pipelines import resolve_outcomes
from mlbsim_models.pipelines.resolve import rollup_eval

pytestmark = pytest.mark.integration

_MODEL = "test_resolve_v1"


@pytest.fixture
def seeded():
    now = dt.datetime.now(dt.UTC)
    with session_scope() as s:
        s.execute(text("TRUNCATE serving.model_versions, serving.model_eval_runs CASCADE"))
        s.execute(text("TRUNCATE warehouse.teams, warehouse.games CASCADE"))
        s.execute(
            text(
                "INSERT INTO warehouse.teams (team_id, abbr, name) VALUES "
                "(1,'AAA','A'),(2,'BBB','B')"
            )
        )
        s.add(ModelVersion(model_id=_MODEL, name="test", kind="logit", version="0", trained_at=now))
        s.flush()

        # 40 finished games; a well-calibrated model that leans on the home team.
        preds = []
        for i in range(40):
            pk = 900000 + i
            home_won = i % 3 != 0  # ~67% home wins
            hs, as_ = (5, 2) if home_won else (2, 5)
            s.add(
                Game(
                    game_pk=pk,
                    season=2026,
                    game_date=dt.date(2026, 6, 1),
                    game_type="R",
                    status="Final",
                    home_team_id=1,
                    away_team_id=2,
                    home_score=hs,
                    away_score=as_,
                )
            )
            preds.append(
                {
                    "game_pk": pk,
                    "model_id": _MODEL,
                    "created_at": now,
                    "is_live": False,
                    "home_win_prob": 0.66 if home_won else 0.60,
                    "away_win_prob": 0.34 if home_won else 0.40,
                }
            )
        # one live prediction + one prediction for an unfinished game: both ignored
        s.add(
            Game(
                game_pk=999001,
                season=2026,
                game_date=dt.date(2026, 6, 2),
                game_type="R",
                status="In Progress",
                home_team_id=1,
                away_team_id=2,
            )
        )
        preds.append(
            {
                "game_pk": 900000,
                "model_id": _MODEL,
                "created_at": now,
                "is_live": True,
                "home_win_prob": 0.7,
                "away_win_prob": 0.3,
            }
        )
        preds.append(
            {
                "game_pk": 999001,
                "model_id": _MODEL,
                "created_at": now,
                "is_live": False,
                "home_win_prob": 0.55,
                "away_win_prob": 0.45,
            }
        )
        s.flush()  # materialise the Game rows before the FK-bearing prediction insert
        s.execute(GamePrediction.__table__.insert(), preds)


def test_resolve_outcomes_scores_only_finished_non_live(seeded):
    summary = resolve_outcomes(since="2026-01-01", model_id=_MODEL)
    assert summary.resolved == 40  # not the live row, not the in-progress game
    assert 0.0 < summary.brier < 0.25
    assert summary.log_loss and summary.log_loss > 0

    with session_scope() as s:
        n = s.execute(select(func.count()).select_from(PredictionOutcome)).scalar_one()
        assert n == 40
        # idempotent: a second pass resolves nothing new
    assert resolve_outcomes(since="2026-01-01", model_id=_MODEL).resolved == 0


def test_rollup_eval_writes_an_eval_run_with_calibration(seeded):
    resolve_outcomes(since="2026-01-01", model_id=_MODEL)
    eval_id = rollup_eval(_MODEL, season=2026)

    with session_scope() as s:
        run = s.get(ModelEvalRun, eval_id)
        assert run is not None
        assert run.metrics_json["model"]["n"] == 40
        assert run.metrics_json["beats_coin_flip_logloss"] is True
        bins = s.execute(
            select(func.count())
            .select_from(CalibrationBin)
            .where(CalibrationBin.eval_id == eval_id)
        ).scalar_one()
        assert bins >= 1
