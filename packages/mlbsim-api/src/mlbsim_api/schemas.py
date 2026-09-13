"""Pydantic response models for the public API."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel


class GameSummary(BaseModel):
    game_pk: int
    season: int | None = None
    game_date: dt.date | None = None
    start_time_utc: dt.datetime | None = None
    status: str | None = None
    is_final: bool = False
    home_team_id: int
    away_team_id: int
    home_abbr: str | None = None
    away_abbr: str | None = None
    home_score: int | None = None
    away_score: int | None = None
    home_win_prob: float | None = None
    away_win_prob: float | None = None
    exp_home_runs: float | None = None
    exp_away_runs: float | None = None
    model_id: str | None = None
    pred_id: int | None = None
    # Verdict — populated once the game is Final and the resolver has run.
    predicted_winner: str | None = None  # "home" | "away"
    actual_winner: str | None = None  # "home" | "away"
    correct: bool | None = None
    brier: float | None = None


class GameDetail(GameSummary):
    park_id: int | None = None
    scheduled_start_utc: dt.datetime | None = None
    home_sp_id: int | None = None
    away_sp_id: int | None = None
    p_extra_innings: float | None = None
    p_shutout_home: float | None = None
    p_shutout_away: float | None = None
    p_one_run_game: float | None = None
    home_score_dist: dict[str, Any] | None = None
    away_score_dist: dict[str, Any] | None = None
    total_runs_dist: dict[str, Any] | None = None
    most_likely_scores: list[dict[str, Any]] | None = None
    factors: dict[str, Any] | None = None


class InningRow(BaseModel):
    inning: int
    side: str
    exp_runs: float
    p_score_1plus: float
    p_score_2plus: float
    p_score_3plus: float
    p_scoreless: float


class InningTable(BaseModel):
    game_pk: int
    source: str  # "simulation" | "none"
    rows: list[InningRow] = []
    p_home_lead_after: list[float] = []


class PlayerPredictionOut(BaseModel):
    player_id: int
    role: str
    model_id: str
    proj: dict[str, Any]
    prob: dict[str, Any]


class GamePlayers(BaseModel):
    game_pk: int
    batters: list[PlayerPredictionOut] = []
    pitchers: list[PlayerPredictionOut] = []


class PlayerSeasonLine(BaseModel):
    season: int
    group: str  # batting | pitching
    split: str
    games: int
    through_date: str | None = None
    stats: dict[str, Any]


class PlayerPredictionHistoryRow(BaseModel):
    pred_id: int
    game_pk: int
    game_date: dt.date | None = None
    model_id: str
    role: str
    created_at: dt.datetime
    proj: dict[str, Any]
    prob: dict[str, Any]


class PlayerCard(BaseModel):
    player_id: int
    full_name: str | None = None
    bats: str | None = None
    throws: str | None = None
    primary_pos: str | None = None
    birth_date: dt.date | None = None
    season_lines: list[PlayerSeasonLine] = []
    recent_predictions: list[PlayerPredictionHistoryRow] = []


class SimulationOut(BaseModel):
    game_pk: int
    sim_run_id: int
    n_sims: int
    engine_version: str
    runtime_ms: int | None = None
    summary: dict[str, Any]


class TeamSummary(BaseModel):
    team_id: int
    abbr: str | None = None
    name: str | None = None
    league: str | None = None
    division: str | None = None
    wins: int | None = None
    losses: int | None = None
    elo: float | None = None
    p_playoffs: float | None = None
    p_division: float | None = None
    p_world_series: float | None = None
    exp_wins: float | None = None
    exp_seed: float | None = None


class StandingRow(BaseModel):
    team_id: int
    abbr: str | None = None
    league: str | None = None
    division: str | None = None
    wins: int
    losses: int
    pct: float
    run_diff: int


class ModelOut(BaseModel):
    model_id: str
    name: str
    kind: str
    version: str
    trained_at: dt.datetime | None = None
    train_end: dt.date | None = None
    metrics: dict[str, Any] = {}


class EvalOut(BaseModel):
    model_id: str
    runs: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []


class PredictionHistoryRow(BaseModel):
    pred_id: int
    game_pk: int
    model_id: str
    created_at: dt.datetime
    game_date: dt.date | None = None
    status: str | None = None
    home_abbr: str | None = None
    away_abbr: str | None = None
    home_win_prob: float
    away_win_prob: float
    predicted_winner: str | None = None  # "home" | "away"
    exp_home_runs: float | None = None
    exp_away_runs: float | None = None
    home_score: int | None = None
    away_score: int | None = None
    actual_winner: str | None = None
    correct: bool | None = None
    brier: float | None = None
    log_loss: float | None = None
    abs_err_total_runs: float | None = None


class AccuracyBucket(BaseModel):
    label: str
    n: int
    correct: int
    accuracy: float | None = None
    brier: float | None = None
    log_loss: float | None = None


class PredictionSummary(BaseModel):
    model_id: str | None = None
    generated_at: dt.datetime
    pending: int
    coin_flip_log_loss: float
    overall: AccuracyBucket
    last_7d: AccuracyBucket
    last_30d: AccuracyBucket
    by_model: list[AccuracyBucket] = []


class ExplanationOut(BaseModel):
    pred_id: int
    game_pk: int
    model_id: str
    home_win_prob: float
    top_factors: list[dict[str, Any]] = []
    explanation: str | None = None


class PlayoffsOut(BaseModel):
    sim_run_id: int | None = None
    season: int | None = None
    as_of: str | None = None
    teams: list[TeamSummary] = []
    series: list[dict[str, Any]] = []
