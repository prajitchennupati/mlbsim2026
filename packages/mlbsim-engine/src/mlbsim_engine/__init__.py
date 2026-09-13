"""Simulation engines.

``game``      plate-appearance-level game simulator (vectorised NumPy)      -- M3
``season``    vectorised remaining-season Monte Carlo                        -- M5
``playoffs``  seeding, tiebreakers, bracket, series simulation               -- M5

Every downstream prediction the platform serves is a read-off from simulations
produced here.
"""

from mlbsim_engine.baserunning import advance_batch
from mlbsim_engine.game import (
    GameSimResult,
    GameState,
    live_win_probability,
    simulate_game,
)
from mlbsim_engine.outcomes import OUTCOMES
from mlbsim_engine.rates import (
    lineup_matrix,
    odds_ratio_matchup,
    rates_from_stat_json,
)
from mlbsim_engine.season import SeasonSetup, SeasonSimResult, simulate_season

__version__ = "0.0.0"

ENGINE_VERSION = "0.1.0"
"""Stamped onto ``serving.simulation_runs.engine_version`` for reproducibility."""

__all__ = [
    "ENGINE_VERSION",
    "OUTCOMES",
    "GameSimResult",
    "GameState",
    "SeasonSetup",
    "SeasonSimResult",
    "advance_batch",
    "lineup_matrix",
    "live_win_probability",
    "odds_ratio_matchup",
    "rates_from_stat_json",
    "simulate_game",
    "simulate_season",
]
