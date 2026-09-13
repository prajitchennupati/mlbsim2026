"""Score stored predictions against finished games and roll up eval metrics."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import insert, select

from mlbsim_core import get_logger, session_scope
from mlbsim_data.models import (
    FINAL_STATUSES,
    CalibrationBin,
    Game,
    GamePrediction,
    ModelEvalRun,
    PredictionOutcome,
)
from mlbsim_models.evaluate.metrics import (
    brier_score,
    calibration_table,
    classification_report,
    log_loss,
)

_log = get_logger(__name__)
_EPS = 1e-6


@dataclass(slots=True)
class ResolveSummary:
    resolved: int
    correct: int
    brier: float | None
    log_loss: float | None


def resolve_outcomes(*, since: str | None = None, model_id: str | None = None) -> ResolveSummary:
    """Fill ``prediction_outcomes`` for predictions whose game is now Final."""
    stmt = (
        select(GamePrediction, Game.home_score, Game.away_score)
        .join(Game, Game.game_pk == GamePrediction.game_pk)
        .outerjoin(PredictionOutcome, PredictionOutcome.pred_id == GamePrediction.pred_id)
        .where(
            GamePrediction.is_live.is_(False),
            Game.status.in_(FINAL_STATUSES),
            Game.home_score.is_not(None),
            Game.away_score.is_not(None),
            PredictionOutcome.pred_id.is_(None),  # not yet resolved
        )
    )
    if since:
        stmt = stmt.where(Game.game_date >= dt.date.fromisoformat(since))
    if model_id:
        stmt = stmt.where(GamePrediction.model_id == model_id)

    rows: list[dict[str, Any]] = []
    now = dt.datetime.now(dt.UTC)
    with session_scope() as s:
        for pred, hs, as_ in s.execute(stmt):
            p = min(max(float(pred.home_win_prob), _EPS), 1.0 - _EPS)
            y = 1 if hs > as_ else 0
            ll = -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
            rows.append(
                {
                    "pred_id": pred.pred_id,
                    "game_pk": pred.game_pk,
                    "resolved_at": now,
                    "actual_home_score": int(hs),
                    "actual_away_score": int(as_),
                    "actual_winner": "home" if y else "away",
                    "brier": round((p - y) ** 2, 5),
                    "log_loss": round(ll, 5),
                    "abs_err_total_runs": None,
                    "correct": (p >= 0.5) == bool(y),
                }
            )
        if rows:
            s.execute(insert(PredictionOutcome), rows)

    summary = ResolveSummary(
        resolved=len(rows),
        correct=sum(1 for r in rows if r["correct"]),
        brier=round(sum(r["brier"] for r in rows) / len(rows), 5) if rows else None,
        log_loss=round(sum(r["log_loss"] for r in rows) / len(rows), 5) if rows else None,
    )
    _log.info("resolve.outcomes", **asdict(summary))
    return summary


def rollup_eval(model_id: str, *, season: int | None = None, split_name: str | None = None) -> int:
    """Aggregate resolved outcomes for a model into a ``model_eval_runs`` row."""
    stmt = (
        select(GamePrediction.home_win_prob, Game.home_score, Game.away_score, Game.game_date)
        .join(Game, Game.game_pk == GamePrediction.game_pk)
        .join(PredictionOutcome, PredictionOutcome.pred_id == GamePrediction.pred_id)
        .where(GamePrediction.model_id == model_id, GamePrediction.is_live.is_(False))
        .order_by(Game.game_date)
    )
    if season is not None:
        stmt = stmt.where(Game.season == season)

    with session_scope() as s:
        recs = s.execute(stmt).all()
        if not recs:
            raise LookupError(f"no resolved predictions for {model_id}")
        p = [float(r.home_win_prob) for r in recs]
        y = [1 if r.home_score > r.away_score else 0 for r in recs]
        dates = [r.game_date for r in recs]
        report = classification_report(y, p)
        coin = [0.5] * len(y)
        coin_ll = log_loss(y, coin)
        metrics = {
            "model": report,
            "baselines": {
                "coin_flip": {"log_loss": coin_ll, "brier": brier_score(y, coin)},
            },
            "beats_coin_flip_logloss": report["log_loss"] < coin_ll,
        }
        name = split_name or (f"resolved_{season}" if season else "resolved_all")
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
        cal = calibration_table(y, p)
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
    _log.info("resolve.rollup", model_id=model_id, split=name, n=len(recs))
    return int(eval_id)
