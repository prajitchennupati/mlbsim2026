"""Train the stacked + calibrated ensemble over the base win models, and predict."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import delete, func, insert, select

from mlbsim_core import get_logger, get_settings, session_scope
from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import FINAL_STATUSES, Game, GamePrediction, ModelVersion
from mlbsim_models.ensemble import StackedWinModel, fit_calibrator, load_calibrator
from mlbsim_models.evaluate.metrics import classification_report, expected_calibration_error

_log = get_logger(__name__)

MODEL_ID = "ensemble_v1"
BASE_MODEL_IDS = ("elo_v1", "direct_v1", "gbm_v1")
_CALIB_FRAC = 0.3


@dataclass(slots=True)
class EnsembleTrainSummary:
    model_id: str
    base_models: list[str]
    n_train: int
    n_calibration: int
    weights: dict[str, float]
    in_sample: dict[str, float]


def _artifact_dir() -> str:
    d = get_settings().data_dir / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _base_prob_rows(
    base_ids: tuple[str, ...], *, before: str | None = None, on_date: str | None = None
) -> list[dict[str, Any]]:
    """One row per game that has a prediction from EVERY base model."""
    latest = (
        select(
            GamePrediction.game_pk,
            GamePrediction.model_id,
            func.max(GamePrediction.created_at).label("mx"),
        )
        .where(GamePrediction.model_id.in_(base_ids), GamePrediction.is_live.is_(False))
        .group_by(GamePrediction.game_pk, GamePrediction.model_id)
        .subquery()
    )
    stmt = (
        select(
            GamePrediction.game_pk,
            GamePrediction.model_id,
            GamePrediction.home_win_prob,
            Game.home_score,
            Game.away_score,
            Game.game_date,
        )
        .join(
            latest,
            (latest.c.game_pk == GamePrediction.game_pk)
            & (latest.c.model_id == GamePrediction.model_id)
            & (latest.c.mx == GamePrediction.created_at),
        )
        .join(Game, Game.game_pk == GamePrediction.game_pk)
        .where(Game.game_type == "R")
    )
    if before:
        stmt = stmt.where(
            Game.game_date < dt.date.fromisoformat(before),
            Game.status.in_(FINAL_STATUSES),
            Game.home_score.is_not(None),
        )
    if on_date:
        stmt = stmt.where(Game.game_date == dt.date.fromisoformat(on_date))

    by_game: dict[int, dict[str, Any]] = {}
    with session_scope() as s:
        for r in s.execute(stmt):
            g = by_game.setdefault(
                r.game_pk,
                {"game_pk": r.game_pk, "game_date": r.game_date, "probs": {}, "y": None},
            )
            g["probs"][r.model_id] = float(r.home_win_prob)
            if r.home_score is not None:
                g["y"] = 1 if r.home_score > r.away_score else 0
    return [g for g in by_game.values() if all(mid in g["probs"] for mid in base_ids)]


def train_ensemble(
    *, train_end: str, base_model_ids: tuple[str, ...] = BASE_MODEL_IDS
) -> EnsembleTrainSummary:
    rows = sorted(_base_prob_rows(base_model_ids, before=train_end), key=lambda r: r["game_date"])
    if len(rows) < 300:
        raise ValueError(f"only {len(rows)} games with all base predictions before {train_end}")

    x = np.array([[r["probs"][m] for m in base_model_ids] for r in rows], dtype=np.float64)
    y = np.array([r["y"] for r in rows], dtype=np.int64)
    cut = int(len(rows) * (1 - _CALIB_FRAC))
    x_fit, y_fit = x[:cut], y[:cut]
    x_cal, y_cal = x[cut:], y[cut:]

    stack = StackedWinModel(list(base_model_ids)).fit(x_fit, y_fit)
    calib = fit_calibrator(stack.predict_proba(x_cal), y_cal)
    calib_kind = "isotonic" if type(calib).__name__ == "IsotonicCalibrator" else "platt"

    final_cal = calib.transform(stack.predict_proba(x_cal))
    report = classification_report(y_cal, final_cal)
    in_sample = {
        "holdout_log_loss": report["log_loss"],
        "holdout_brier": report["brier"],
        "holdout_auc": report["roc_auc"],
        "holdout_ece": report["ece"],
        "holdout_ece_uncalibrated": expected_calibration_error(y_cal, stack.predict_proba(x_cal)),
    }

    adir = _artifact_dir()
    stack.save(f"{adir}/{MODEL_ID}.stack.joblib")
    calib.save(f"{adir}/{MODEL_ID}.calibrator.joblib")

    with session_scope() as s:
        upsert(
            s,
            ModelVersion,
            [
                {
                    "model_id": MODEL_ID,
                    "name": "Stacked + isotonic-calibrated ensemble",
                    "kind": "ensemble",
                    "version": "1",
                    "trained_at": dt.datetime.now(dt.UTC),
                    "train_end": dt.date.fromisoformat(train_end),
                    "hyperparams_json": {
                        "base_models": list(base_model_ids),
                        "meta": "LogisticRegression on base logits",
                        "calibrator": calib_kind,
                        "calib_frac": _CALIB_FRAC,
                    },
                    "artifact_uri": adir,
                    "metrics_json": {
                        "in_sample": {k: round(v, 5) for k, v in in_sample.items()},
                        "weights": stack.weights(),
                    },
                    "notes": (
                        f"Meta-learner over base win probs; {calib_kind}-calibrated on a holdout."
                    ),
                }
            ],
            index_elements=["model_id"],
        )

    summary = EnsembleTrainSummary(
        model_id=MODEL_ID,
        base_models=list(base_model_ids),
        n_train=cut,
        n_calibration=len(rows) - cut,
        weights={k: round(v, 4) for k, v in stack.weights().items()},
        in_sample={k: round(v, 5) for k, v in in_sample.items()},
    )
    _log.info("ensemble.train", model_id=MODEL_ID, n_train=cut, **summary.in_sample)
    return summary


def predict_ensemble_date(
    date: str, *, base_model_ids: tuple[str, ...] = BASE_MODEL_IDS, rebuild: bool = True
) -> dict[str, Any]:
    rows = _base_prob_rows(base_model_ids, on_date=date)
    if not rows:
        return {"date": date, "model_id": MODEL_ID, "written": 0, "games": 0}

    with session_scope() as s:
        if s.get(ModelVersion, MODEL_ID) is None:
            return {
                "date": date,
                "model_id": MODEL_ID,
                "written": 0,
                "games": len(rows),
                "skipped": "ensemble not trained (run `mlbsim ensemble train`)",
            }

    adir = get_settings().data_dir / "artifacts"
    stack = StackedWinModel.load(adir / f"{MODEL_ID}.stack.joblib")
    calib = load_calibrator(adir / f"{MODEL_ID}.calibrator.joblib")

    x = np.array([[r["probs"][m] for m in base_model_ids] for r in rows], dtype=np.float64)
    p = calib.transform(stack.predict_proba(x))
    now = dt.datetime.now(dt.UTC)
    game_pks = [r["game_pk"] for r in rows]

    out_rows = [
        {
            "game_pk": r["game_pk"],
            "model_id": MODEL_ID,
            "created_at": now,
            "is_live": False,
            "home_win_prob": round(float(pi), 5),
            "away_win_prob": round(1.0 - float(pi), 5),
            "factors": {
                "base_probs": {m: round(r["probs"][m], 5) for m in base_model_ids},
                "stack_weights": {k: round(v, 4) for k, v in stack.weights().items()},
            },
        }
        for r, pi in zip(rows, p, strict=True)
    ]

    with session_scope() as s:
        if rebuild:
            s.execute(
                delete(GamePrediction).where(
                    GamePrediction.model_id == MODEL_ID,
                    GamePrediction.game_pk.in_(game_pks),
                )
            )
        s.execute(insert(GamePrediction), out_rows)

    _log.info("ensemble.predict", date=date, written=len(out_rows))
    return {"date": date, "model_id": MODEL_ID, "written": len(out_rows), "games": len(rows)}
