"""Render a natural-language explanation of a stored game prediction."""

from __future__ import annotations

from typing import Any

from sqlalchemy import desc, select

from mlbsim_core import session_scope
from mlbsim_data.models import Game, GamePrediction, Team
from mlbsim_models.explain import render_explanation

_DEFAULT_MODEL = "direct_v1"


def explain_game(
    game_pk: int, *, model_id: str = _DEFAULT_MODEL, backend: str = "template"
) -> dict[str, Any]:
    with session_scope() as s:
        game = s.get(Game, game_pk)
        if game is None:
            raise LookupError(f"game {game_pk} not ingested")
        names: dict[int, str] = dict(s.execute(select(Team.team_id, Team.abbr)).tuples().all())
        pred = s.scalar(
            select(GamePrediction)
            .where(GamePrediction.model_id == model_id, GamePrediction.game_pk == game_pk)
            .order_by(desc(GamePrediction.created_at))
            .limit(1)
        )
        if pred is None:
            raise LookupError(f"no {model_id} prediction for game {game_pk}")
        home = names.get(game.home_team_id, str(game.home_team_id))
        away = names.get(game.away_team_id, str(game.away_team_id))

    factors = (pred.factors or {}).get("top_factors", [])
    context = {
        "home": home,
        "away": away,
        "date": game.game_date.isoformat() if game.game_date else None,
        "home_win_prob": float(pred.home_win_prob),
        "exp_home_runs": float(pred.exp_home_runs) if pred.exp_home_runs is not None else None,
        "exp_away_runs": float(pred.exp_away_runs) if pred.exp_away_runs is not None else None,
    }
    return {
        "context": context,
        "factors": factors,
        "explanation": render_explanation(context, factors, backend=backend),
    }
