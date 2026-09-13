"""Shared query helpers for the routers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from mlbsim_api.schemas import GameSummary
from mlbsim_data.models import Game, GamePrediction, PredictionOutcome, Team, is_game_final

# Preference order when a game has predictions from several models.
PREFERRED_MODELS = ("ensemble_v1", "direct_v1", "elo_v1")


def predicted_winner(pred: GamePrediction) -> str:
    """The side the model favours ("home" | "away"); ties break to home."""
    return "home" if float(pred.home_win_prob) >= 0.5 else "away"


def build_game_summary(
    g: Game,
    abbr: Mapping[int, str],
    pred: GamePrediction | None,
    outcome: PredictionOutcome | None,
) -> GameSummary:
    """The fields every games/teams endpoint returns for one game — factored out
    so `is_final`, the verdict fields, etc. can't drift between endpoints the
    way `/teams/{abbr}/schedule` once did (it built its own GameSummary by hand
    and simply never got them)."""
    return GameSummary(
        game_pk=g.game_pk,
        season=g.season,
        game_date=g.game_date,
        start_time_utc=g.scheduled_start_utc,
        status=g.status,
        is_final=is_game_final(g.status),
        home_team_id=g.home_team_id,
        away_team_id=g.away_team_id,
        home_abbr=abbr.get(g.home_team_id),
        away_abbr=abbr.get(g.away_team_id),
        home_score=g.home_score,
        away_score=g.away_score,
        home_win_prob=float(pred.home_win_prob) if pred else None,
        away_win_prob=float(pred.away_win_prob) if pred else None,
        exp_home_runs=float(pred.exp_home_runs)
        if pred and pred.exp_home_runs is not None
        else None,
        exp_away_runs=float(pred.exp_away_runs)
        if pred and pred.exp_away_runs is not None
        else None,
        model_id=pred.model_id if pred else None,
        pred_id=pred.pred_id if pred else None,
        predicted_winner=predicted_winner(pred) if pred else None,
        actual_winner=outcome.actual_winner if outcome else None,
        correct=outcome.correct if outcome else None,
        brier=float(outcome.brier) if outcome and outcome.brier is not None else None,
    )


def outcomes_by_pred_id(session: Session, pred_ids: Iterable[int]) -> dict[int, PredictionOutcome]:
    """Bulk-load resolved outcomes so game lists avoid an N+1 query."""
    ids = [p for p in pred_ids if p is not None]
    if not ids:
        return {}
    return {
        o.pred_id: o
        for o in session.scalars(
            select(PredictionOutcome).where(PredictionOutcome.pred_id.in_(ids))
        )
    }


def team_abbr_map(session: Session) -> dict[int, str]:
    return dict(session.execute(select(Team.team_id, Team.abbr)).tuples().all())


def team_by_abbr(session: Session, abbr: str) -> Team | None:
    return session.scalar(select(Team).where(Team.abbr == abbr.upper()))


def latest_prediction(
    session: Session, game_pk: int, *, models: tuple[str, ...] = PREFERRED_MODELS
) -> GamePrediction | None:
    """Newest non-live prediction for a game, honouring model preference order."""
    rows = list(
        session.scalars(
            select(GamePrediction)
            .where(GamePrediction.game_pk == game_pk, GamePrediction.is_live.is_(False))
            .order_by(desc(GamePrediction.created_at))
        )
    )
    if not rows:
        return None
    by_model: dict[str, GamePrediction] = {}
    for r in rows:
        by_model.setdefault(r.model_id, r)
    for m in models:
        if m in by_model:
            return by_model[m]
    return rows[0]


def render_explanation(context: dict[str, Any], factors: list[dict[str, Any]]) -> str:
    """Tiny deterministic template — keeps the API free of the ML dependency chain."""
    home = context.get("home", "the home team")
    away = context.get("away", "the away team")
    p = float(context["home_win_prob"])
    favourite, prob = (home, p) if p >= 0.5 else (away, 1.0 - p)
    parts = [f"The model gives {favourite} a {round(100 * prob)}% chance to win"]
    if context.get("exp_home_runs") is not None:
        parts.append(
            f", with an expected score of {context['exp_home_runs']:.1f}-"
            f"{context['exp_away_runs']:.1f}"
        )
    parts.append(". ")
    for_fav = [f for f in factors if (f.get("favours") == "home") == (favourite == home)]
    against = [f for f in factors if f not in for_fav]
    if for_fav:
        parts.append("The main reasons are " + ", ".join(f["label"] for f in for_fav[:3]))
    if against:
        parts.append(f", partly offset by {against[0]['label']}")
    parts.append(".")
    return "".join(parts)
