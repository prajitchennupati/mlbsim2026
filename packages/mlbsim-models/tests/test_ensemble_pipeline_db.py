"""Ensemble train/predict + llm_experiments recording against a live database."""

from __future__ import annotations

import datetime as dt
import random

import pytest
from sqlalchemy import select, text

from mlbsim_core import session_scope
from mlbsim_data.models import (
    Game,
    GamePrediction,
    LlmExperiment,
    ModelVersion,
)
from mlbsim_models.llm.experiment import compare_and_verdict, record_experiment
from mlbsim_models.pipelines import predict_ensemble_date, train_ensemble

pytestmark = pytest.mark.integration

_TEAMS = [(1, "AAA"), (2, "BBB"), (3, "CCC"), (4, "DDD")]
_TRAIN_END = "2024-08-15"
_PREDICT_DATE = "2024-08-20"
_TABLES = ["games", "teams"]


def _register(s, model_id, kind):
    s.execute(
        text(
            "INSERT INTO serving.model_versions (model_id, name, kind, version, trained_at) "
            f"VALUES ('{model_id}', '{model_id}', '{kind}', '1', now()) "
            "ON CONFLICT (model_id) DO NOTHING"
        )
    )


@pytest.fixture
def base_predictions():
    rng = random.Random(3)
    with session_scope() as s:
        s.execute(text("TRUNCATE warehouse.games, warehouse.teams CASCADE"))
        s.execute(text("TRUNCATE serving.model_versions CASCADE"))
        s.execute(
            text(
                "INSERT INTO warehouse.teams (team_id, abbr, name) VALUES "
                + ",".join(f"({t},'{a}','{a}')" for t, a in _TEAMS)
            )
        )
        _register(s, "elo_v1", "elo")
        _register(s, "direct_v1", "logit")
        _register(s, "gbm_v1", "gbm")

        games, preds = [], []
        pk = 700000
        day = dt.date(2024, 5, 1)
        for _ in range(500):
            h, a = rng.sample([1, 2, 3, 4], 2)
            p_true = rng.uniform(0.30, 0.70)
            home_won = rng.random() < p_true
            games.append(
                {
                    "game_pk": pk,
                    "season": 2024,
                    "game_date": day,
                    "game_type": "R",
                    "status": "Final",
                    "home_team_id": h,
                    "away_team_id": a,
                    "home_score": 5 if home_won else 3,
                    "away_score": 3 if home_won else 5,
                    "dh_game_num": 1,
                    "is_doubleheader": False,
                    "scheduled_innings": 9,
                }
            )
            # elo: noisier; direct: sharper but slightly biased high; gbm: sharp, unbiased
            elo_p = min(max(p_true + rng.gauss(0, 0.10), 0.02), 0.98)
            direct_p = min(max(p_true + 0.05 + rng.gauss(0, 0.05), 0.02), 0.98)
            gbm_p = min(max(p_true + rng.gauss(0, 0.06), 0.02), 0.98)
            for mid, p in (("elo_v1", elo_p), ("direct_v1", direct_p), ("gbm_v1", gbm_p)):
                preds.append(
                    {
                        "game_pk": pk,
                        "model_id": mid,
                        "created_at": dt.datetime(2024, 5, 1, tzinfo=dt.UTC),
                        "is_live": False,
                        "home_win_prob": round(p, 5),
                        "away_win_prob": round(1 - p, 5),
                    }
                )
            pk += 1
            day += dt.timedelta(days=1) if pk % 3 == 0 else dt.timedelta(0)
        s.execute(Game.__table__.insert(), games)
        s.execute(GamePrediction.__table__.insert(), preds)
    yield


def test_train_and_predict_ensemble(base_predictions):
    summary = train_ensemble(train_end=_TRAIN_END)
    assert summary.model_id == "ensemble_v1"
    assert summary.n_train > 0 and summary.n_calibration > 0
    assert set(summary.weights) == {"intercept", "elo_v1", "direct_v1", "gbm_v1"}
    assert summary.in_sample["holdout_log_loss"] < 0.75  # sane

    with session_scope() as s:
        mv = s.get(ModelVersion, "ensemble_v1")
        assert mv is not None and mv.kind == "ensemble"
        assert mv.metrics_json["weights"]

    out = predict_ensemble_date(_PREDICT_DATE)
    assert out["written"] == out["games"] >= 0

    if out["written"]:
        with session_scope() as s:
            row = s.scalar(
                select(GamePrediction).where(GamePrediction.model_id == "ensemble_v1").limit(1)
            )
            assert 0.0 < float(row.home_win_prob) < 1.0
            assert set(row.factors["base_probs"]) == {"elo_v1", "direct_v1", "gbm_v1"}


def test_record_llm_experiment(base_predictions):
    import numpy as np

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    base = np.clip(y * 0.6 + 0.2 + rng.normal(0, 0.1, 200), 0.02, 0.98)
    llm = np.full(200, 0.5)
    cmp = compare_and_verdict(llm, base, y, baseline_name="direct_v1")
    exp_id = record_experiment(
        base_model="claude-sonnet-5",
        task="win_prob_zeroshot",
        train_spec={"n_test": 200, "prompt": "game_card v1"},
        metrics=cmp,
        verdict=cmp["verdict"],
    )
    with session_scope() as s:
        row = s.get(LlmExperiment, exp_id)
        assert row is not None
        assert row.task == "win_prob_zeroshot"
        assert "does NOT beat" in row.verdict
        assert row.metrics_json["baseline"]["log_loss"] < row.metrics_json["llm"]["log_loss"]
