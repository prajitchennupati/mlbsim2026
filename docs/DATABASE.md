# Database Design

PostgreSQL (Supabase). Two logical layers in one instance: schema `warehouse` (ingested
baseball facts) and schema `serving` (precomputed, versioned predictions the API reads).
Migrations via **Alembic** (`db/migrations/`). Model artifacts live in object storage and
are referenced by `serving.model_versions.artifact_uri` — never stored in the DB.

## Conventions

- All timestamps are UTC (`timestamptz`).
- Point-in-time tables carry a `source_ts` (when the fact became known).
- Prediction tables are **append-only and immutable**; a new prediction is a new row with
  a new `model_id` / `created_at`, never an update.
- `base_state` is encoded 0–7 as a 3-bit mask (bit 0 = runner on 1B, bit 1 = 2B,
  bit 2 = 3B).
- Large event tables (`plate_appearances`, `pitches`) carry a `season` column and a
  `season` index. Native **list-partitioning by `season` is planned but deferred** to a
  dedicated migration once it can be validated against a live database (unpartitioned
  handles the ~2M-row scale comfortably in the meantime).

---

## Schema `warehouse`

### Dimensions

| Table | Key columns |
|---|---|
| `teams` | `team_id` (MLBAM) PK · `abbr` · `name` · `league` · `division` · `park_id` |
| `parks` | `park_id` PK · `name` · `lat` · `lon` · `altitude_ft` · `roof_type` · `orientation_deg` · `lf_dist` `cf_dist` `rf_dist` · `pf_runs_3yr` · `pf_hr_3yr` |
| `players` | `player_id` (MLBAM) PK · `retro_id` · `fg_id` · `bbref_id` · `full_name` · `bats` · `throws` · `birth_date` · `primary_pos` · `debut_date` |
| `seasons` | `season` PK · `start_date` · `end_date` · `n_playoff_teams` · `rules_json` (ghost-runner rule, roster size, schedule length) |

### Game facts

| Table | Key / notable columns |
|---|---|
| `games` | `game_pk` PK · `season` · `game_date` · `game_type` (R/F/D/L/W/S) · `home_team_id` · `away_team_id` · `park_id` · `scheduled_start_utc` · `status` · `home_score` · `away_score` · `n_innings` · `is_doubleheader` · `dh_game_num` · `home_sp_id` · `away_sp_id` · `winning_pitcher_id` · `losing_pitcher_id` · `save_pitcher_id` |
| `game_weather` | `game_pk` PK · `forecast_ts` · `temp_f` · `wind_mph` · `wind_dir_deg` · `humidity` · `condition` · `observed_temp_f` (post-hoc only) |
| `game_probables` | (`game_pk`, `team_id`) PK · `probable_pitcher_id` · `source_ts` — point-in-time |
| `lineups` | (`game_pk`, `team_id`, `batting_order`) PK · `player_id` · `position` · `source` (projected/actual) · `source_ts` |
| `plate_appearances` | `pa_id` PK · `game_pk` · `inning` · `half` · `batter_id` · `pitcher_id` · `batting_order` · `outs_start` · `base_state_start` · `base_state_end` · `outs_end` · `event_type` · `rbi` · `runs_on_play` · `is_sac` · `times_through_order` · `pitcher_pitch_no` · `leverage_index` · `wpa` — **partitioned by season** |
| `pitches` | `pitch_id` PK · `pa_id` · `seq` · `pitch_type` · `release_speed` · `spin_rate` · `plate_x` · `plate_z` · `balls` · `strikes` · `description` · `zone` — **partitioned by season**; optional/large (Statcast) |
| `batted_balls` | `pa_id` PK · `launch_speed` · `launch_angle` · `hit_distance` · `xba` · `xwoba` · `is_barrel` · `is_hardhit` |

### Aggregates (materialised for speed; derived from `plate_appearances` where possible)

| Table | Key columns |
|---|---|
| `batting_game_logs` | (`player_id`, `game_pk`) PK · `team_id` · `pa` `ab` `h` `d2b` `t3b` `hr` `r` `rbi` `bb` `hbp` `so` `sb` `cs` `tb` |
| `pitching_game_logs` | (`player_id`, `game_pk`) PK · `team_id` · `outs` `bf` `h` `r` `er` `bb` `hbp` `so` `hr` `pitches` · `is_start` · `got_win` · `got_save` · `quality_start` |
| `team_game_logs` | (`team_id`, `game_pk`) PK · `runs_for` · `runs_against` · `is_home` · `won` |
| `player_season_stats` | (`player_id`, `season`, `group`, `split`) PK · `through_date` (NULL = full season) · `games` · `stat_json` (JSONB: counting + rate stats — `woba`, `wrc_plus`, `fip`, `xfip`, `siera`, `barrel_pct`, …). `group` ∈ {batting, pitching}; `split` ∈ {all, vs_L, vs_R, home, away, last15, last30} |
| `team_season_stats` | (`team_id`, `season`, `split`) PK · `through_date` · `stat_json` (JSONB: run environment, OPS, wOBA, bullpen ERA/FIP, team FIP/xFIP/SIERA, DRS/OAA) |
| `transactions` | `txn_id` PK · `player_id` · `team_id` · `type` (IL/activate/callup/option/trade) · `effective_date` · `resolution_date` — as-of roster reconstruction |
| `elo_ratings` | (`entity_type`, `entity_id`, `as_of_date`) PK · `rating` · `rating_sp_adj` · `games_played` — append-only, strictly chronological |

### Feature store

| Table | Key / notable columns |
|---|---|
| `feature_sets` | `feature_set_id` PK · `name` · `version` · `spec_json` · `created_at` |
| `game_features` | (`game_pk`, `feature_set_id`, `side`) PK · `as_of_ts` · `features` (JSONB) · `data_snapshot_hash` · `computed_at` — `side` ∈ {home, away, diff} |
| `player_game_features` | (`player_id`, `game_pk`, `feature_set_id`) PK · `as_of_ts` · `features` (JSONB) · `data_snapshot_hash` · `computed_at` |

`data_snapshot_hash` = hash of contributing source row IDs + max `source_ts`. Two runs of
the builder for the same `(entity, feature_set_id, as_of_ts)` must produce an identical
hash; CI enforces this.

---

## Schema `serving`

### Models

| Table | Key / notable columns |
|---|---|
| `model_versions` | `model_id` PK · `name` · `kind` (elo/logit/poisson/nb/gbm/nn/event/sim/ensemble/llm) · `version` · `git_sha` · `trained_at` · `train_start` · `train_end` · `feature_set_id` · `hyperparams_json` · `artifact_uri` · `metrics_json` · `notes` |

A scoring guard refuses to produce predictions for a game whose `game_date <= train_end`
of the model being used.

### Predictions (append-only, immutable)

| Table | Key / notable columns |
|---|---|
| `game_predictions` | `pred_id` PK · `game_pk` · `model_id` · `created_at` · `is_live` · `game_state_json` (nullable) · `home_win_prob` · `away_win_prob` · `exp_home_runs` · `exp_away_runs` · `home_score_dist` (JSONB array) · `away_score_dist` · `total_runs_dist` · `run_diff_dist` · `p_extra_innings` · `p_shutout_home` · `p_shutout_away` · `p_one_run_game` · `most_likely_scores` (JSONB) · `factors` (JSONB, SHAP-derived) |
| `inning_predictions` | (`pred_id`, `inning`, `team_id`) PK · `exp_runs` · `p_score_1plus` · `p_score_2plus` · `p_score_3plus` · `runs_dist` (JSONB) · `p_lead_after` · `p_scoreless` |
| `player_predictions` | `pred_id` PK · `game_pk` · `player_id` · `model_id` · `role` (bat/pitch) · `created_at` · `proj_json` (point estimates) · `prob_json` (P(1+H), P(2+H), P(HR), P(5+K), …) · `dist_json` (per-stat pmf) |

### Simulation

| Table | Key / notable columns |
|---|---|
| `simulation_runs` | `sim_run_id` PK · `scope` (game/season/series) · `target_id` · `model_id` · `n_sims` · `seed` · `engine_version` · `runtime_ms` · `params_json` · `created_at` |
| `game_sim_results` | (`sim_run_id`, `game_pk`) PK · win prob · score grid (JSONB) · player stat-line distributions (JSONB) |
| `season_sim_team_results` | (`sim_run_id`, `team_id`) PK · `p_playoffs` · `p_division` · `p_wildcard` · `p_bye` · `p_pennant` · `p_world_series` · `exp_wins` · `exp_losses` · `exp_seed` · `win_dist` (JSONB) |
| `series_predictions` | (`sim_run_id`, `round`, `team_a_id`, `team_b_id`) PK · `best_of` (3/5/7) · `p_a_wins` · `exp_games` · `p_sweep` · `game_count_dist` (JSONB: `{2:..,3:..}` for best-of-3, `{3,4,5}` for best-of-5, `{4,5,6,7}` for best-of-7) |

### Standings, outcomes, evaluation

| Table | Key / notable columns |
|---|---|
| `standings` | (`season`, `snapshot_date`, `team_id`) PK · `w` · `l` · `pct` · `gb` · `rs` · `ra` · `streak` · `source` (actual/projected) |
| `prediction_outcomes` | `pred_id` PK · `game_pk` · `resolved_at` · `actual_home_score` · `actual_away_score` · `actual_winner` · `brier` · `log_loss` · `abs_err_total_runs` · `correct` (bool) |
| `model_eval_runs` | `eval_id` PK · `model_id` · `split_name` · `period_start` · `period_end` · `metrics_json` · `created_at` |
| `calibration_bins` | (`eval_id`, `bin_lower`) PK · `bin_upper` · `n` · `mean_pred` · `mean_actual` |
| `prediction_explanations` | (`pred_id`, `method`) PK · `payload_json` · `text` · `model_id` · `created_at` — `method` ∈ {shap, llm} |
| `llm_experiments` | `exp_id` PK · `base_model` · `task` · `train_spec_json` · `metrics_json` · `verdict` · `created_at` |

---

## Indexing & partitioning

- `warehouse.games (game_date)`, `warehouse.games (season)`
- `warehouse.plate_appearances (game_pk)`, `warehouse.plate_appearances (batter_id, game_pk)`,
  `warehouse.plate_appearances (pitcher_id, game_pk)`
- `warehouse.batting_game_logs (player_id)`, `warehouse.pitching_game_logs (player_id)`
- `warehouse.game_features (game_pk, feature_set_id)`
- `serving.game_predictions (game_pk, model_id, created_at DESC)`
- partial: `serving.game_predictions (game_pk) WHERE is_live`
- `serving.season_sim_team_results (sim_run_id)`
- (Planned) list-partition `warehouse.plate_appearances` and `warehouse.pitches` by
  `season` — deferred, see the conventions note above.

## Data volume (rough)

- ~2,430 games/season x ~76 PA ≈ 185k PA rows/season; 2015–2025 ≈ 2.0M PA rows.
- Statcast ~700k pitches/season — kept as Parquet in object storage, only modelled
  columns loaded into `pitches` / `batted_balls`.
- Comfortable on Supabase free / low tiers.
