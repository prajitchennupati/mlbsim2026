"""Prediction history (vs actuals), rolling accuracy, per-prediction explanations."""

from __future__ import annotations

import datetime as dt
import math

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from mlbsim_api._common import (
    predicted_winner,
    render_explanation,
    team_abbr_map,
    team_by_abbr,
)
from mlbsim_api.deps import get_session
from mlbsim_api.schemas import (
    AccuracyBucket,
    ExplanationOut,
    PredictionHistoryRow,
    PredictionSummary,
)
from mlbsim_data.models import Game, GamePrediction, PredictionOutcome

router = APIRouter(prefix="/predictions", tags=["predictions"])

_COIN_FLIP_LOG_LOSS = math.log(2)


@router.get("/history", response_model=list[PredictionHistoryRow])
def prediction_history(
    *,
    date: dt.date | None = Query(None, description="single game date"),
    date_from: dt.date | None = Query(None),
    date_to: dt.date | None = Query(None),
    team: str | None = Query(None),
    model: str | None = Query(None),
    resolved: bool | None = Query(None, description="filter to graded / ungraded picks"),
    limit: int = Query(200, le=1000),
    s: Session = Depends(get_session),
) -> list[PredictionHistoryRow]:
    abbr = team_abbr_map(s)
    stmt = (
        select(GamePrediction, PredictionOutcome, Game)
        .join(Game, Game.game_pk == GamePrediction.game_pk)
        .outerjoin(PredictionOutcome, PredictionOutcome.pred_id == GamePrediction.pred_id)
        .where(GamePrediction.is_live.is_(False))
        .order_by(desc(Game.game_date), desc(GamePrediction.created_at))
        .limit(limit)
    )
    if date:
        stmt = stmt.where(Game.game_date == date)
    if date_from:
        stmt = stmt.where(Game.game_date >= date_from)
    if date_to:
        stmt = stmt.where(Game.game_date <= date_to)
    if model:
        stmt = stmt.where(GamePrediction.model_id == model)
    if resolved is True:
        stmt = stmt.where(PredictionOutcome.pred_id.is_not(None))
    elif resolved is False:
        stmt = stmt.where(PredictionOutcome.pred_id.is_(None))
    if team:
        t = team_by_abbr(s, team)
        if t is None:
            raise HTTPException(404, f"unknown team {team!r}")
        stmt = stmt.where((Game.home_team_id == t.team_id) | (Game.away_team_id == t.team_id))

    out: list[PredictionHistoryRow] = []
    for pred, outcome, game in s.execute(stmt):
        out.append(
            PredictionHistoryRow(
                pred_id=pred.pred_id,
                game_pk=pred.game_pk,
                model_id=pred.model_id,
                created_at=pred.created_at,
                game_date=game.game_date,
                status=game.status,
                home_abbr=abbr.get(game.home_team_id),
                away_abbr=abbr.get(game.away_team_id),
                home_win_prob=float(pred.home_win_prob),
                away_win_prob=float(pred.away_win_prob),
                predicted_winner=predicted_winner(pred),
                exp_home_runs=float(pred.exp_home_runs) if pred.exp_home_runs is not None else None,
                exp_away_runs=float(pred.exp_away_runs) if pred.exp_away_runs is not None else None,
                home_score=game.home_score,
                away_score=game.away_score,
                actual_winner=outcome.actual_winner if outcome else None,
                correct=outcome.correct if outcome else None,
                brier=float(outcome.brier) if outcome and outcome.brier is not None else None,
                log_loss=float(outcome.log_loss)
                if outcome and outcome.log_loss is not None
                else None,
                abs_err_total_runs=float(outcome.abs_err_total_runs)
                if outcome and outcome.abs_err_total_runs is not None
                else None,
            )
        )
    return out


def _bucket(label: str, rows: list[tuple[bool, float | None, float | None]]) -> AccuracyBucket:
    n = len(rows)
    if not n:
        return AccuracyBucket(label=label, n=0, correct=0)
    briers = [b for _, b, _ in rows if b is not None]
    lls = [ll for _, _, ll in rows if ll is not None]
    hits = sum(1 for c, _, _ in rows if c)
    return AccuracyBucket(
        label=label,
        n=n,
        correct=hits,
        accuracy=hits / n,
        brier=sum(briers) / len(briers) if briers else None,
        log_loss=sum(lls) / len(lls) if lls else None,
    )


@router.get("/summary", response_model=PredictionSummary)
def prediction_summary(
    model: str | None = Query(None, description="restrict to one model_id"),
    s: Session = Depends(get_session),
) -> PredictionSummary:
    """Rolling scorecard: overall / last-7d / last-30d accuracy, plus per-model."""
    today = dt.date.today()
    graded = (
        select(
            GamePrediction.model_id,
            Game.game_date,
            PredictionOutcome.correct,
            PredictionOutcome.brier,
            PredictionOutcome.log_loss,
        )
        .join(Game, Game.game_pk == GamePrediction.game_pk)
        .join(PredictionOutcome, PredictionOutcome.pred_id == GamePrediction.pred_id)
        .where(GamePrediction.is_live.is_(False))
    )
    if model:
        graded = graded.where(GamePrediction.model_id == model)

    all_rows = [
        (mid, gdate, bool(correct), _f(brier), _f(ll))
        for mid, gdate, correct, brier, ll in s.execute(graded)
    ]

    def since(days: int) -> list[tuple[bool, float | None, float | None]]:
        cut = today - dt.timedelta(days=days)
        return [(c, b, ll) for _, d, c, b, ll in all_rows if d is not None and d >= cut]

    overall = [(c, b, ll) for _, _, c, b, ll in all_rows]
    by_model: list[AccuracyBucket] = []
    for mid in sorted({r[0] for r in all_rows}):
        by_model.append(_bucket(mid, [(c, b, ll) for m, _, c, b, ll in all_rows if m == mid]))

    pending_stmt = (
        select(GamePrediction.pred_id)
        .join(Game, Game.game_pk == GamePrediction.game_pk)
        .outerjoin(PredictionOutcome, PredictionOutcome.pred_id == GamePrediction.pred_id)
        .where(GamePrediction.is_live.is_(False), PredictionOutcome.pred_id.is_(None))
    )
    if model:
        pending_stmt = pending_stmt.where(GamePrediction.model_id == model)
    pending = len(list(s.scalars(pending_stmt)))

    return PredictionSummary(
        model_id=model,
        generated_at=dt.datetime.now(dt.UTC),
        pending=pending,
        coin_flip_log_loss=_COIN_FLIP_LOG_LOSS,
        overall=_bucket("all-time", overall),
        last_7d=_bucket("last 7 days", since(7)),
        last_30d=_bucket("last 30 days", since(30)),
        by_model=by_model,
    )


def _f(x: object) -> float | None:
    return float(x) if x is not None else None  # type: ignore[arg-type]


@router.get("/{pred_id}/explanation", response_model=ExplanationOut)
def prediction_explanation(pred_id: int, s: Session = Depends(get_session)) -> ExplanationOut:
    pred = s.get(GamePrediction, pred_id)
    if pred is None:
        raise HTTPException(404, f"prediction {pred_id} not found")
    game = s.get(Game, pred.game_pk)
    abbr = team_abbr_map(s)
    factors = (pred.factors or {}).get("top_factors", [])
    context = {
        "home": abbr.get(game.home_team_id) if game else None,
        "away": abbr.get(game.away_team_id) if game else None,
        "home_win_prob": float(pred.home_win_prob),
        "exp_home_runs": float(pred.exp_home_runs) if pred.exp_home_runs is not None else None,
        "exp_away_runs": float(pred.exp_away_runs) if pred.exp_away_runs is not None else None,
    }
    return ExplanationOut(
        pred_id=pred_id,
        game_pk=pred.game_pk,
        model_id=pred.model_id,
        home_win_prob=float(pred.home_win_prob),
        top_factors=factors,
        explanation=render_explanation(context, factors) if factors else None,
    )
