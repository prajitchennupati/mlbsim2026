"""Train the direct models (logistic win prob + Poisson run models) and register them."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from mlbsim_core import get_logger, get_settings, session_scope
from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import Game, GameFeature, ModelVersion
from mlbsim_features import DIFF_FEATURE_ORDER, feature_vector
from mlbsim_models.direct import RunsPoisson, WinLogit
from mlbsim_models.evaluate.metrics import brier_score, classification_report

_log = get_logger(__name__)

MODEL_ID = "direct_v1"


@dataclass(slots=True)
class TrainSummary:
    model_id: str
    n_train: int
    train_start: str
    train_end: str
    in_sample: dict[str, float]
    artifact_dir: str


def _artifact_dir() -> str:
    d = get_settings().data_dir / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _load_training_rows(
    train_end: str, train_start: str | None, feature_set_id: str
) -> list[dict[str, Any]]:
    stmt = (
        select(
            GameFeature.features,
            Game.home_score,
            Game.away_score,
            Game.game_date,
        )
        .join(Game, Game.game_pk == GameFeature.game_pk)
        .where(
            GameFeature.feature_set_id == feature_set_id,
            GameFeature.side == "diff",
            Game.game_type == "R",
            Game.home_score.is_not(None),
            Game.game_date < dt.date.fromisoformat(train_end),
        )
        .order_by(Game.game_date)
    )
    if train_start:
        stmt = stmt.where(Game.game_date >= dt.date.fromisoformat(train_start))
    with session_scope() as s:
        return [
            {
                "x": feature_vector(r.features),
                "home_won": 1 if r.home_score > r.away_score else 0,
                "home_runs": r.home_score,
                "away_runs": r.away_score,
            }
            for r in s.execute(stmt)
        ]


def train_direct_models(
    *,
    train_end: str,
    train_start: str | None = None,
    feature_set_id: str = "fs_v1",
) -> TrainSummary:
    rows = _load_training_rows(train_end, train_start, feature_set_id)
    if len(rows) < 100:
        raise ValueError(
            f"only {len(rows)} training games with features before {train_end} — "
            "run `mlbsim features build` first"
        )

    x = [r["x"] for r in rows]
    y_win = [r["home_won"] for r in rows]
    y_home = [r["home_runs"] for r in rows]
    y_away = [r["away_runs"] for r in rows]
    names = list(DIFF_FEATURE_ORDER)

    win = WinLogit(names).fit(x, y_win)
    runs_home = RunsPoisson(names).fit(x, y_home)
    runs_away = RunsPoisson(names).fit(x, y_away)

    adir = _artifact_dir()
    win.save(f"{adir}/{MODEL_ID}.win_logit.joblib")
    runs_home.save(f"{adir}/{MODEL_ID}.runs_home.joblib")
    runs_away.save(f"{adir}/{MODEL_ID}.runs_away.joblib")

    p = win.predict_proba(x)
    report = classification_report(y_win, p)
    mu_home = runs_home.predict_mu(x)
    in_sample = {
        "win_log_loss": report["log_loss"],
        "win_brier": report["brier"],
        "win_auc": report["roc_auc"],
        "runs_home_brier_vs_mean": brier_score(
            [1 if r > 4.5 else 0 for r in y_home], [min(m / 9.0, 1.0) for m in mu_home]
        ),
    }
    now = dt.datetime.now(dt.UTC)
    with session_scope() as s:
        upsert(
            s,
            ModelVersion,
            [
                {
                    "model_id": MODEL_ID,
                    "name": "Direct: logistic win prob + Poisson runs",
                    "kind": "logit",
                    "version": "1",
                    "git_sha": os.environ.get("GIT_SHA"),
                    "trained_at": now,
                    "train_start": dt.date.fromisoformat(train_start) if train_start else None,
                    "train_end": dt.date.fromisoformat(train_end),
                    "feature_set_id": feature_set_id,
                    "hyperparams_json": {
                        "win": {"model": "LogisticRegression", "C": win.c},
                        "runs": {"model": "GLM/Poisson", "sides": ["home", "away"]},
                        "features": names,
                    },
                    "artifact_uri": adir,
                    "metrics_json": {"in_sample": in_sample, "n_train": len(rows)},
                    "notes": "Consumes fs_v1 diff features; independent Poisson run sides.",
                }
            ],
            index_elements=["model_id"],
        )

    summary = TrainSummary(
        model_id=MODEL_ID,
        n_train=len(rows),
        train_start=train_start or "(earliest available)",
        train_end=train_end,
        in_sample={k: round(v, 5) for k, v in in_sample.items()},
        artifact_dir=adir,
    )
    _log.info("train.direct", model_id=MODEL_ID, n_train=len(rows), **summary.in_sample)
    return summary
