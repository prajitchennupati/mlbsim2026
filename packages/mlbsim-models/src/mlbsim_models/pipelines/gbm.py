"""Train and predict the gradient-boosted win model (``gbm_v1``).

A non-linear base model for the ensemble stack, on the same ``fs_v1`` diff
features as ``direct_v1``. Win probability only — the Poisson run models stay
with ``direct_v1``.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, insert, select

from mlbsim_core import get_logger, session_scope
from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import Game, GamePrediction, ModelVersion
from mlbsim_features import DIFF_FEATURE_ORDER, feature_vector
from mlbsim_models.direct import GBMWinModel
from mlbsim_models.evaluate.metrics import classification_report
from mlbsim_models.pipelines.predict import _diff_features_for_date
from mlbsim_models.pipelines.train import _artifact_dir, _load_training_rows

_log = get_logger(__name__)

MODEL_ID = "gbm_v1"


@dataclass(slots=True)
class GBMTrainSummary:
    model_id: str
    n_train: int
    train_end: str
    in_sample: dict[str, float]


def _model_path() -> str:
    return f"{_artifact_dir()}/{MODEL_ID}.win_gbm.joblib"


def train_gbm(
    *, train_end: str, train_start: str | None = None, feature_set_id: str = "fs_v1"
) -> GBMTrainSummary:
    rows = _load_training_rows(train_end, train_start, feature_set_id)
    if len(rows) < 100:
        raise ValueError(
            f"only {len(rows)} training games with features before {train_end} — "
            "run `mlbsim features build` first"
        )

    names = list(DIFF_FEATURE_ORDER)
    x = [r["x"] for r in rows]
    y = [r["home_won"] for r in rows]
    model = GBMWinModel(names).fit(x, y)
    model.save(_model_path())

    report = classification_report(y, model.predict_proba(x))
    in_sample = {
        "win_log_loss": report["log_loss"],
        "win_brier": report["brier"],
        "win_auc": report["roc_auc"],
    }
    with session_scope() as s:
        upsert(
            s,
            ModelVersion,
            [
                {
                    "model_id": MODEL_ID,
                    "name": "Gradient-boosted win probability",
                    "kind": "gbm",
                    "version": "1",
                    "git_sha": os.environ.get("GIT_SHA"),
                    "trained_at": dt.datetime.now(dt.UTC),
                    "train_start": dt.date.fromisoformat(train_start) if train_start else None,
                    "train_end": dt.date.fromisoformat(train_end),
                    "feature_set_id": feature_set_id,
                    "hyperparams_json": {"model": "HistGradientBoosting", **model.params},
                    "artifact_uri": _artifact_dir(),
                    "metrics_json": {"in_sample": in_sample, "n_train": len(rows)},
                    "notes": "HistGradientBoosting on fs_v1 diff features; ensemble base model.",
                }
            ],
            index_elements=["model_id"],
        )

    summary = GBMTrainSummary(
        model_id=MODEL_ID,
        n_train=len(rows),
        train_end=train_end,
        in_sample={k: round(v, 5) for k, v in in_sample.items()},
    )
    _log.info("train.gbm", model_id=MODEL_ID, n_train=len(rows), **summary.in_sample)
    return summary


def predict_gbm_date(
    date: str, *, feature_set_id: str = "fs_v1", rebuild: bool = True
) -> dict[str, Any]:
    """Write one ``gbm_v1`` ``game_predictions`` row per game on ``date``."""
    with session_scope() as s:
        game_pks = list(
            s.scalars(
                select(Game.game_pk).where(
                    Game.game_date == dt.date.fromisoformat(date), Game.game_type == "R"
                )
            )
        )
    if not game_pks:
        return {"date": date, "model_id": MODEL_ID, "games": 0, "written": 0}

    feats = _diff_features_for_date(date, feature_set_id)
    model = GBMWinModel.load(_model_path())
    now = dt.datetime.now(dt.UTC)

    rows: list[dict[str, Any]] = []
    for game_pk in game_pks:
        diff = feats.get(game_pk)
        if diff is None:
            continue
        p_home = float(model.predict_proba([feature_vector(diff)])[0])
        rows.append(
            {
                "game_pk": game_pk,
                "model_id": MODEL_ID,
                "created_at": now,
                "is_live": False,
                "home_win_prob": round(p_home, 5),
                "away_win_prob": round(1.0 - p_home, 5),
                "factors": {"win_source": "gbm"},
            }
        )

    with session_scope() as s:
        if rebuild:
            s.execute(
                delete(GamePrediction).where(
                    GamePrediction.model_id == MODEL_ID,
                    GamePrediction.game_pk.in_(game_pks),
                )
            )
        if rows:
            s.execute(insert(GamePrediction), rows)

    out = {"date": date, "model_id": MODEL_ID, "games": len(game_pks), "written": len(rows)}
    _log.info("predict.gbm", **out)
    return out
