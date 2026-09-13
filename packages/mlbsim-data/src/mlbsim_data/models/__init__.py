"""SQLAlchemy ORM models — the source of truth for the database schema.

Alembic autogenerates migrations from ``mlbsim_core.Base.metadata``, so importing
this package must register every table on that metadata object.

Scope by milestone (see ``docs/DATABASE.md``):
  M0  dimensions
  M1  game facts, play-by-play, aggregate stat tables, transactions, Elo
  M2+ feature store, model registry, prediction + simulation tables (``serving``)
"""

from mlbsim_data.models.dims import Park, Player, Season, Team
from mlbsim_data.models.features import FeatureSet, GameFeature
from mlbsim_data.models.games import Game, GameProbable, GameWeather, Lineup
from mlbsim_data.models.misc import EloRating, Transaction
from mlbsim_data.models.pbp import BattedBall, Pitch, PlateAppearance
from mlbsim_data.models.serving import (
    CalibrationBin,
    GamePrediction,
    GameSimResultRow,
    LlmExperiment,
    ModelEvalRun,
    ModelVersion,
    PlayerPrediction,
    PredictionOutcome,
    SeasonSimTeamResult,
    SeriesPrediction,
    SimulationRun,
)
from mlbsim_data.models.stats import (
    BattingGameLog,
    PitchingGameLog,
    PlayerSeasonStat,
    TeamGameLog,
    TeamSeasonStat,
)

__all__ = [
    "BattedBall",
    "BattingGameLog",
    "CalibrationBin",
    "EloRating",
    "FeatureSet",
    "Game",
    "GameFeature",
    "GamePrediction",
    "GameProbable",
    "GameSimResultRow",
    "GameWeather",
    "Lineup",
    "LlmExperiment",
    "ModelEvalRun",
    "ModelVersion",
    "Park",
    "Pitch",
    "PitchingGameLog",
    "PlateAppearance",
    "Player",
    "PlayerPrediction",
    "PlayerSeasonStat",
    "PredictionOutcome",
    "Season",
    "SeasonSimTeamResult",
    "SeriesPrediction",
    "SimulationRun",
    "Team",
    "TeamGameLog",
    "TeamSeasonStat",
    "Transaction",
]
