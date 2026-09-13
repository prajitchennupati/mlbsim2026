"""API routers, all mounted under ``/api/v1``."""

from mlbsim_api.routers import (
    games,
    models,
    players,
    playoffs,
    predictions,
    simulations,
    standings,
    teams,
)

ROUTERS = (
    games.router,
    teams.router,
    players.router,
    standings.router,
    playoffs.router,
    simulations.router,
    models.router,
    predictions.router,
)

__all__ = ["ROUTERS"]
