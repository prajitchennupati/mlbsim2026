"""Model evaluation: proper scoring rules, calibration, baselines, backtesting."""

from mlbsim_models.evaluate.backtest import (
    WalkForwardSplit,
    split_rows_by_date,
    walk_forward_splits,
)
from mlbsim_models.evaluate.baselines import (
    coin_flip_prob,
    elo_probs,
    home_team_prob,
)
from mlbsim_models.evaluate.metrics import (
    accuracy,
    brier_score,
    calibration_table,
    classification_report,
    expected_calibration_error,
    log_loss,
    mae,
    poisson_deviance,
    rmse,
    roc_auc,
)
from mlbsim_models.evaluate.projections import projection_error

__all__ = [
    "WalkForwardSplit",
    "accuracy",
    "brier_score",
    "calibration_table",
    "classification_report",
    "coin_flip_prob",
    "elo_probs",
    "expected_calibration_error",
    "home_team_prob",
    "log_loss",
    "mae",
    "poisson_deviance",
    "projection_error",
    "rmse",
    "roc_auc",
    "split_rows_by_date",
    "walk_forward_splits",
]
