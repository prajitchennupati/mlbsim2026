"""A read-only health snapshot of the warehouse + serving schema."""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import func, select

from mlbsim_core import get_logger, session_scope
from mlbsim_data.models import (
    Game,
    GamePrediction,
    ModelVersion,
    PredictionOutcome,
    SimulationRun,
)

_log = get_logger(__name__)


def _age_days(d: dt.date | dt.datetime | None) -> float | None:
    if d is None:
        return None
    if isinstance(d, dt.datetime):
        now = dt.datetime.now(d.tzinfo or dt.UTC)
        return round((now - d).total_seconds() / 86_400, 2)
    return round((dt.date.today() - d).days, 2)


def pipeline_status() -> dict[str, Any]:
    """Return a dict of freshness / backlog indicators for monitoring + the CLI."""
    with session_scope() as s:
        last_game_date = s.execute(select(func.max(Game.game_date))).scalar_one_or_none()
        last_final_date = s.execute(
            select(func.max(Game.game_date)).where(Game.home_score.is_not(None))
        ).scalar_one_or_none()
        n_games = s.execute(select(func.count()).select_from(Game)).scalar_one()

        model_rows = s.execute(
            select(
                GamePrediction.model_id,
                func.max(GamePrediction.created_at),
                func.count(),
            ).group_by(GamePrediction.model_id)
        ).all()
        models = {
            mid: {
                "last_prediction_at": ts.isoformat() if ts else None,
                "age_days": _age_days(ts),
                "n_predictions": int(n),
            }
            for mid, ts, n in model_rows
        }

        sim_rows = s.execute(
            select(SimulationRun.scope, func.max(SimulationRun.created_at), func.count()).group_by(
                SimulationRun.scope
            )
        ).all()
        sims = {
            scope: {
                "last_run_at": ts.isoformat() if ts else None,
                "age_days": _age_days(ts),
                "n_runs": int(n),
            }
            for scope, ts, n in sim_rows
        }

        registered = s.execute(
            select(ModelVersion.model_id, ModelVersion.kind, ModelVersion.trained_at)
        ).all()
        registry = {
            mid: {
                "kind": kind,
                "trained_at": ta.isoformat() if ta else None,
                "age_days": _age_days(ta),
            }
            for mid, kind, ta in registered
        }

        unresolved = s.execute(
            select(func.count())
            .select_from(GamePrediction)
            .join(Game, Game.game_pk == GamePrediction.game_pk)
            .outerjoin(PredictionOutcome, PredictionOutcome.pred_id == GamePrediction.pred_id)
            .where(
                GamePrediction.is_live.is_(False),
                Game.home_score.is_not(None),
                PredictionOutcome.pred_id.is_(None),
            )
        ).scalar_one()

    status = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "warehouse": {
            "n_games": int(n_games),
            "last_game_date": last_game_date.isoformat() if last_game_date else None,
            "last_final_game_date": last_final_date.isoformat() if last_final_date else None,
            "days_since_last_final": _age_days(last_final_date),
        },
        "predictions": models,
        "simulations": sims,
        "registry": registry,
        "unresolved_predictions": int(unresolved),
    }
    _log.info("flow.status", unresolved=int(unresolved), n_models=len(models))
    return status
