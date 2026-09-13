"""Model pipelines: read the warehouse, run a model, write the serving schema."""

from mlbsim_models.pipelines.elo import build_elo, predict_elo_date
from mlbsim_models.pipelines.ensemble import predict_ensemble_date, train_ensemble
from mlbsim_models.pipelines.evaluate import evaluate_model
from mlbsim_models.pipelines.explain import explain_game
from mlbsim_models.pipelines.gbm import predict_gbm_date, train_gbm
from mlbsim_models.pipelines.live import live_win_prob_for_game, update_live_games
from mlbsim_models.pipelines.player_predict import predict_players_for_game
from mlbsim_models.pipelines.predict import predict_date
from mlbsim_models.pipelines.projections import marcel_for_player, store_marcel_projections
from mlbsim_models.pipelines.resolve import resolve_outcomes, rollup_eval
from mlbsim_models.pipelines.season import build_season_setup, simulate_season_from_db
from mlbsim_models.pipelines.train import train_direct_models

__all__ = [
    "build_elo",
    "build_season_setup",
    "evaluate_model",
    "explain_game",
    "live_win_prob_for_game",
    "marcel_for_player",
    "predict_date",
    "predict_elo_date",
    "predict_ensemble_date",
    "predict_gbm_date",
    "predict_players_for_game",
    "resolve_outcomes",
    "rollup_eval",
    "simulate_season_from_db",
    "store_marcel_projections",
    "train_direct_models",
    "train_ensemble",
    "train_gbm",
    "update_live_games",
]
