/**
 * Response shapes for the mlbplayoffs2026 API (mirrors `mlbsim_api/schemas.py`).
 * Regenerate the exact OpenAPI types with `npm run gen:types` -> lib/api-types.ts.
 */

export type Side = "home" | "away";

export interface GameSummary {
  game_pk: number;
  season?: number | null;
  game_date?: string | null;
  start_time_utc?: string | null;
  status?: string | null;
  is_final: boolean;
  home_team_id: number;
  away_team_id: number;
  home_abbr?: string | null;
  away_abbr?: string | null;
  home_score?: number | null;
  away_score?: number | null;
  home_win_prob?: number | null;
  away_win_prob?: number | null;
  exp_home_runs?: number | null;
  exp_away_runs?: number | null;
  model_id?: string | null;
  pred_id?: number | null;
  predicted_winner?: Side | null;
  actual_winner?: Side | null;
  correct?: boolean | null;
  brier?: number | null;
}

export interface Factor {
  feature: string;
  label: string;
  logit_contribution: number;
  favours: Side | string;
  prob_shift: number;
}

export interface GameDetail extends GameSummary {
  park_id?: number | null;
  scheduled_start_utc?: string | null;
  home_sp_id?: number | null;
  away_sp_id?: number | null;
  p_extra_innings?: number | null;
  p_shutout_home?: number | null;
  p_shutout_away?: number | null;
  p_one_run_game?: number | null;
  home_score_dist?: Record<string, number> | null;
  away_score_dist?: Record<string, number> | null;
  total_runs_dist?: Record<string, number> | null;
  most_likely_scores?: { home: number; away: number; p: number }[] | null;
  factors?: { top_factors?: Factor[]; [k: string]: unknown } | null;
}

export interface InningRow {
  inning: number;
  side: Side;
  exp_runs: number;
  p_score_1plus: number;
  p_score_2plus: number;
  p_score_3plus: number;
  p_scoreless: number;
}

export interface InningTable {
  game_pk: number;
  source: string;
  rows: InningRow[];
  p_home_lead_after: number[];
}

export interface PlayerPredictionOut {
  player_id: number;
  role: "bat" | "pitch";
  model_id: string;
  proj: Record<string, number>;
  prob: Record<string, number>;
}

export interface GamePlayers {
  game_pk: number;
  batters: PlayerPredictionOut[];
  pitchers: PlayerPredictionOut[];
}

export interface SimulationOut {
  game_pk: number;
  sim_run_id: number;
  n_sims: number;
  engine_version: string;
  runtime_ms?: number | null;
  summary: Record<string, unknown>;
}

export interface TeamSummary {
  team_id: number;
  abbr?: string | null;
  name?: string | null;
  league?: string | null;
  division?: string | null;
  wins?: number | null;
  losses?: number | null;
  elo?: number | null;
  p_playoffs?: number | null;
  p_division?: number | null;
  p_world_series?: number | null;
  exp_wins?: number | null;
  exp_seed?: number | null;
}

export interface StandingRow {
  team_id: number;
  abbr?: string | null;
  league?: string | null;
  division?: string | null;
  wins: number;
  losses: number;
  pct: number;
  run_diff: number;
}

export interface ModelOut {
  model_id: string;
  name: string;
  kind: string;
  version: string;
  trained_at?: string | null;
  train_end?: string | null;
  metrics: Record<string, unknown>;
}

export interface EvalOut {
  model_id: string;
  runs: {
    eval_id: number;
    split_name: string;
    period_start: string | null;
    period_end: string | null;
    metrics: Record<string, unknown>;
    created_at: string;
  }[];
  calibration: {
    eval_id: number;
    bin_lower: number;
    bin_upper: number;
    n: number;
    mean_pred: number;
    mean_actual: number;
  }[];
}

export interface PredictionHistoryRow {
  pred_id: number;
  game_pk: number;
  model_id: string;
  created_at: string;
  game_date?: string | null;
  status?: string | null;
  home_abbr?: string | null;
  away_abbr?: string | null;
  home_win_prob: number;
  away_win_prob: number;
  predicted_winner?: Side | null;
  exp_home_runs?: number | null;
  exp_away_runs?: number | null;
  home_score?: number | null;
  away_score?: number | null;
  actual_winner?: Side | null;
  correct?: boolean | null;
  brier?: number | null;
  log_loss?: number | null;
  abs_err_total_runs?: number | null;
}

export interface AccuracyBucket {
  label: string;
  n: number;
  correct: number;
  accuracy?: number | null;
  brier?: number | null;
  log_loss?: number | null;
}

export interface PredictionSummary {
  model_id?: string | null;
  generated_at: string;
  pending: number;
  coin_flip_log_loss: number;
  overall: AccuracyBucket;
  last_7d: AccuracyBucket;
  last_30d: AccuracyBucket;
  by_model: AccuracyBucket[];
}

export interface ExplanationOut {
  pred_id: number;
  game_pk: number;
  model_id: string;
  home_win_prob: number;
  top_factors: Factor[];
  explanation?: string | null;
}

export interface SeriesRow {
  round: string;
  best_of: number;
  p_sweep: number;
  exp_games: number;
  game_count_dist: Record<string, number>;
}

export interface PlayoffsOut {
  sim_run_id?: number | null;
  season?: number | null;
  as_of?: string | null;
  teams: TeamSummary[];
  series: SeriesRow[];
}
