"""Score a model's stored predictions against actual outcomes vs named baselines."""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import insert, select

from mlbsim_core import get_logger, session_scope
from mlbsim_data.models import CalibrationBin, Game, GamePrediction, ModelEvalRun
from mlbsim_models.evaluate.metrics import (
    brier_score,
    calibration_table,
    classification_report,
    log_loss,
)

_log = get_logger(__name__)


def _baseline_block(y: list[int], p: list[float]) -> dict[str, float]:
    return {"log_loss": log_loss(y, p), "brier": brier_score(y, p), "n": float(len(y))}


def evaluate_model(
    model_id: str,
    *,
    season: int | None = None,
    start: str | None = None,
    end: str | None = None,
    split_name: str | None = None,
    home_win_rate: float = 0.54,
    persist: bool = True,
) -> dict[str, Any]:
    """Evaluate ``model_id`` over a period; optionally write model_eval_runs + bins."""
    stmt = (
        select(
            GamePrediction.home_win_prob,
            Game.home_score,
            Game.away_score,
            Game.game_date,
        )
        .join(Game, Game.game_pk == GamePrediction.game_pk)
        .where(
            GamePrediction.model_id == model_id,
            GamePrediction.is_live.is_(False),
            Game.home_score.is_not(None),
        )
        .order_by(Game.game_date)
    )
    if season is not None:
        stmt = stmt.where(Game.season == season)
    if start:
        stmt = stmt.where(Game.game_date >= dt.date.fromisoformat(start))
    if end:
        stmt = stmt.where(Game.game_date <= dt.date.fromisoformat(end))

    with session_scope() as s:
        rows = s.execute(stmt).all()
        p = [float(r.home_win_prob) for r in rows]
        y = [1 if r.home_score > r.away_score else 0 for r in rows]
        dates = [r.game_date for r in rows]

        if not rows:
            raise LookupError(f"no stored predictions for {model_id} in the requested period")

        report = classification_report(y, p)
        cal = calibration_table(y, p)
        metrics = {
            "model": report,
            "baselines": {
                "coin_flip": _baseline_block(y, [0.5] * len(y)),
                "home_team": _baseline_block(y, [home_win_rate] * len(y)),
            },
            "beats_coin_flip_logloss": report["log_loss"] < log_loss(y, [0.5] * len(y)),
            "beats_home_team_logloss": report["log_loss"] < log_loss(y, [home_win_rate] * len(y)),
        }

        name = split_name or (f"season_{season}" if season else f"{start or 'all'}_{end or 'all'}")
        if persist:
            eval_id = s.execute(
                insert(ModelEvalRun)
                .values(
                    model_id=model_id,
                    split_name=name,
                    period_start=min(dates),
                    period_end=max(dates),
                    metrics_json=metrics,
                    created_at=dt.datetime.now(dt.UTC),
                )
                .returning(ModelEvalRun.eval_id)
            ).scalar_one()
            if cal:
                s.execute(
                    insert(CalibrationBin),
                    [
                        {
                            "eval_id": eval_id,
                            **{
                                k: b[k]
                                for k in ("bin_lower", "bin_upper", "n", "mean_pred", "mean_actual")
                            },
                        }
                        for b in cal
                    ],
                )
            metrics["eval_id"] = eval_id

    metrics["split_name"] = name
    _log.info(
        "model.evaluate",
        model_id=model_id,
        split=name,
        n=report["n"],
        log_loss=round(report["log_loss"], 4),
        brier=round(report["brier"], 4),
        auc=round(report["roc_auc"], 4),
    )
    return metrics
