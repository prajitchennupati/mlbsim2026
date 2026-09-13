"""Serving-layer tables (``serving`` schema): model registry, predictions, outcomes.

These are written only by the pipeline and read only by the API. Predictions are
append-only and immutable — a new prediction is a new row carrying its own
``model_id`` and ``created_at``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    ForeignKey,
    Index,
    Numeric,
    SmallInteger,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from mlbsim_core.db import Base
from mlbsim_data.models._types import ts, ts_opt

_S = "serving"
_W = "warehouse"


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = {"schema": _S}

    model_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # e.g. "elo_v1"
    name: Mapped[str] = mapped_column(String(64))
    # elo | logit | poisson | nb | gbm | sim | ensemble
    kind: Mapped[str] = mapped_column(String(16))
    version: Mapped[str] = mapped_column(String(32))
    git_sha: Mapped[str | None] = mapped_column(String(40))
    trained_at: Mapped[ts]
    train_start: Mapped[dt.date | None]
    train_end: Mapped[dt.date | None]
    feature_set_id: Mapped[str | None] = mapped_column(String(64))
    hyperparams_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    artifact_uri: Mapped[str | None] = mapped_column(String(400))
    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    notes: Mapped[str | None] = mapped_column(String(1000))


class GamePrediction(Base):
    __tablename__ = "game_predictions"
    __table_args__ = (
        Index("ix_game_predictions_game_model", "game_pk", "model_id", "created_at"),
        Index("ix_game_predictions_live", "game_pk", postgresql_where=text("is_live")),
        {"schema": _S},
    )

    pred_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    game_pk: Mapped[int] = mapped_column(BigInteger, ForeignKey(f"{_W}.games.game_pk"))
    model_id: Mapped[str] = mapped_column(String(64), ForeignKey(f"{_S}.model_versions.model_id"))
    created_at: Mapped[ts]
    is_live: Mapped[bool] = mapped_column(Boolean, default=False)
    game_state_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    home_win_prob: Mapped[float] = mapped_column(Numeric(6, 5))
    away_win_prob: Mapped[float] = mapped_column(Numeric(6, 5))
    exp_home_runs: Mapped[float | None] = mapped_column(Numeric(5, 3))
    exp_away_runs: Mapped[float | None] = mapped_column(Numeric(5, 3))

    home_score_dist: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    away_score_dist: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    total_runs_dist: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    run_diff_dist: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    p_extra_innings: Mapped[float | None] = mapped_column(Numeric(6, 5))
    p_shutout_home: Mapped[float | None] = mapped_column(Numeric(6, 5))
    p_shutout_away: Mapped[float | None] = mapped_column(Numeric(6, 5))
    p_one_run_game: Mapped[float | None] = mapped_column(Numeric(6, 5))
    most_likely_scores: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    factors: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class PredictionOutcome(Base):
    __tablename__ = "prediction_outcomes"
    __table_args__ = {"schema": _S}

    pred_id: Mapped[int] = mapped_column(
        BigInteger,
        # a resolved outcome is a child of its prediction: rebuilding predictions
        # (delete + re-insert) should carry the stale outcome away with it.
        ForeignKey(f"{_S}.game_predictions.pred_id", ondelete="CASCADE"),
        primary_key=True,
    )
    game_pk: Mapped[int] = mapped_column(BigInteger)
    resolved_at: Mapped[ts_opt]
    actual_home_score: Mapped[int | None] = mapped_column(SmallInteger)
    actual_away_score: Mapped[int | None] = mapped_column(SmallInteger)
    actual_winner: Mapped[str | None] = mapped_column(String(4))  # home | away
    brier: Mapped[float | None] = mapped_column(Numeric(7, 5))
    log_loss: Mapped[float | None] = mapped_column(Numeric(9, 5))
    abs_err_total_runs: Mapped[float | None] = mapped_column(Numeric(6, 3))
    correct: Mapped[bool | None] = mapped_column(Boolean)


class PlayerPrediction(Base):
    __tablename__ = "player_predictions"
    __table_args__ = (
        Index("ix_player_predictions_game_model", "game_pk", "model_id"),
        Index("ix_player_predictions_player", "player_id"),
        {"schema": _S},
    )

    pred_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    game_pk: Mapped[int] = mapped_column(BigInteger, ForeignKey(f"{_W}.games.game_pk"))
    player_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(f"{_W}.players.player_id"))
    model_id: Mapped[str] = mapped_column(String(64), ForeignKey(f"{_S}.model_versions.model_id"))
    role: Mapped[str] = mapped_column(String(5))  # bat | pitch
    created_at: Mapped[ts]
    proj_json: Mapped[dict[str, Any]] = mapped_column(JSONB)  # point estimates
    prob_json: Mapped[dict[str, Any]] = mapped_column(JSONB)  # P(1+H), P(HR), P(5+K), ...
    dist_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class ModelEvalRun(Base):
    __tablename__ = "model_eval_runs"
    __table_args__ = (
        Index("ix_model_eval_runs_model", "model_id", "split_name"),
        {"schema": _S},
    )

    eval_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(64), ForeignKey(f"{_S}.model_versions.model_id"))
    split_name: Mapped[str] = mapped_column(String(48))  # e.g. "test_2024", "wf_2024-07"
    period_start: Mapped[dt.date | None]
    period_end: Mapped[dt.date | None]
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[ts]


class CalibrationBin(Base):
    __tablename__ = "calibration_bins"
    __table_args__ = {"schema": _S}

    eval_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{_S}.model_eval_runs.eval_id"), primary_key=True
    )
    bin_lower: Mapped[float] = mapped_column(Numeric(4, 3), primary_key=True)
    bin_upper: Mapped[float] = mapped_column(Numeric(4, 3))
    n: Mapped[int] = mapped_column(BigInteger)
    mean_pred: Mapped[float] = mapped_column(Numeric(6, 5))
    mean_actual: Mapped[float] = mapped_column(Numeric(6, 5))


class SimulationRun(Base):
    __tablename__ = "simulation_runs"
    __table_args__ = (
        Index("ix_simulation_runs_scope_target", "scope", "target_id"),
        {"schema": _S},
    )

    sim_run_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    scope: Mapped[str] = mapped_column(String(8))  # game | season | series
    target_id: Mapped[int | None] = mapped_column(BigInteger)  # game_pk / season / ...
    model_id: Mapped[str | None] = mapped_column(String(64))
    engine_version: Mapped[str] = mapped_column(String(16))
    n_sims: Mapped[int] = mapped_column(BigInteger)
    seed: Mapped[int] = mapped_column(BigInteger)
    runtime_ms: Mapped[int | None] = mapped_column(BigInteger)
    params_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[ts]


class GameSimResultRow(Base):
    __tablename__ = "game_sim_results"
    __table_args__ = {"schema": _S}

    sim_run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{_S}.simulation_runs.sim_run_id"), primary_key=True
    )
    game_pk: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{_W}.games.game_pk"), primary_key=True
    )
    home_win_prob: Mapped[float] = mapped_column(Numeric(6, 5))
    exp_home_runs: Mapped[float] = mapped_column(Numeric(5, 3))
    exp_away_runs: Mapped[float] = mapped_column(Numeric(5, 3))
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSONB)


class SeasonSimTeamResult(Base):
    __tablename__ = "season_sim_team_results"
    __table_args__ = {"schema": _S}

    sim_run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{_S}.simulation_runs.sim_run_id"), primary_key=True
    )
    team_id: Mapped[int] = mapped_column(ForeignKey(f"{_W}.teams.team_id"), primary_key=True)
    p_playoffs: Mapped[float] = mapped_column(Numeric(6, 5))
    p_division: Mapped[float] = mapped_column(Numeric(6, 5))
    p_wildcard: Mapped[float] = mapped_column(Numeric(6, 5))
    p_bye: Mapped[float] = mapped_column(Numeric(6, 5))
    p_pennant: Mapped[float] = mapped_column(Numeric(6, 5))
    p_world_series: Mapped[float] = mapped_column(Numeric(6, 5))
    exp_wins: Mapped[float] = mapped_column(Numeric(5, 2))
    exp_losses: Mapped[float] = mapped_column(Numeric(5, 2))
    exp_seed: Mapped[float | None] = mapped_column(Numeric(4, 2))
    exp_postseason_game_wins: Mapped[float] = mapped_column(Numeric(5, 3))
    win_dist: Mapped[dict[str, Any]] = mapped_column(JSONB)


class SeriesPrediction(Base):
    __tablename__ = "series_predictions"
    __table_args__ = {"schema": _S}

    sim_run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{_S}.simulation_runs.sim_run_id"), primary_key=True
    )
    round: Mapped[str] = mapped_column(String(4), primary_key=True)  # WC | DS | LCS | WS
    best_of: Mapped[int] = mapped_column(SmallInteger)
    p_sweep: Mapped[float] = mapped_column(Numeric(6, 5))
    exp_games: Mapped[float] = mapped_column(Numeric(4, 3))
    game_count_dist: Mapped[dict[str, Any]] = mapped_column(JSONB)


class LlmExperiment(Base):
    __tablename__ = "llm_experiments"
    __table_args__ = {"schema": _S}

    exp_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    base_model: Mapped[str] = mapped_column(
        String(80)
    )  # e.g. "claude-sonnet-5", "llama-3.1-8b+lora"
    task: Mapped[str] = mapped_column(String(64))  # e.g. "win_prob_zeroshot", "win_prob_lora"
    train_spec_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    verdict: Mapped[str] = mapped_column(String(2000))
    created_at: Mapped[ts]
