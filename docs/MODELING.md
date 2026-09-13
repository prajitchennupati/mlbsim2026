# Modelling

## 1. Philosophy

Optimise for correct methodology, not flashy numbers. Specifically:

- Every model is measured **out-of-sample** with **time-ordered** validation.
- Every model has a **named baseline** it must beat; if it does not, that is the
  documented result.
- Probabilistic outputs must be **calibrated**, not just discriminative.
- The LLM component earns its place only where it genuinely helps; if fine-tuning does not
  improve predictive performance, the write-up says so plainly.

## 2. The four learned components of the simulator

### 2.1 Event model — PA outcome multinomial

Predicts `P(outcome | batter, pitcher, context)` over
`{field_out, K, BB, HBP, 1B, 2B, 3B, HR}`.

| Stage | Method | Rationale |
|---|---|---|
| Baseline | **log5 / odds-ratio** (Tango): combine batter rate, pitcher rate, league rate per outcome, renormalise | Low variance on small matchup samples; interpretable; historically hard to beat |
| Upgrade 1 | **Gradient-boosted multinomial** (LightGBM) on batter/pitcher rate features + context (park, platoon, times-through-order, temperature, altitude) | Captures interactions log5 cannot; still fast |
| Upgrade 2 | **Neural net with player embeddings** — learned batter and pitcher vectors → softmax over 8 outcomes | Learns latent style (high-K power bat vs sinkerballer); embeddings double as the "player representation" deliverable and feed similarity features |

Context features: park factor, batter/pitcher handedness platoon, times-through-order
penalty, temperature, altitude, catcher framing (later), umpire (later). Count-level
modelling is out of scope for v1 (PA-level only).

Talent rates that feed log5 come from the **projection model** (§2.4), not season-to-date
raw rates, so early-season predictions are not dominated by noise.

### 2.2 Baserunning model

Given `(event_type, outs, base_state)`, produce a distribution over
`(new_base_state, runs_scored, outs_added)`. Baseline is the **empirical transition
matrix** estimated from the 2015+ play-by-play (with light smoothing for rare cells).
Optional upgrade: condition on batter speed / hit trajectory. Errors and other rare events
are folded into the `field_out` / advancement tables empirically.

**PA data source.** The MLB Stats API game feed (`/api/v1.1/game/{pk}/feed/live`,
`liveData.plays.allPlays`) carries every needed field — event type, pre/post base-out
state, RBI, running score, pitch sequence — for **2015–present**, which is exactly the
locked full-PA backfill window (ADR 0001). Retrosheet is therefore **not** on the M1
critical path; it becomes relevant only if history is later extended before 2015, and is
tracked as an optional "extend history" task. Retrosheet's copyright notice is still
displayed (see `docs/DATA_SOURCES.md`).

### 2.3 Bullpen model

Two parts:

1. **Usage** — rule-based reliever-entry model: starter pulled on pitch-count / times-
   through-order / runs-allowed thresholds calibrated to league behaviour; leverage-based
   selection of the next arm (closer in save situations, etc.); back-to-back-day
   availability from recent game logs.
2. **Performance** — each reliever gets his own event-model rates (via the same log5 / NN
   machinery). Unknown September call-ups fall back to a role-and-team prior.

### 2.4 Projection model — player true-talent rates

| Stage | Method | Rationale |
|---|---|---|
| Baseline | **Marcel** — 3-year weighted rates (5/4/3), regression to the mean, age curve | The honest baseline; beating it reliably is a real result |
| Upgrade | **Hierarchical Bayes** partial pooling — player rate shrunk toward position/league priors with sample-size-aware strength | Handles rookies / call-ups without hand-tuned regression constants; yields posterior draws that become simulation uncertainty |
| External check | **Steamer / ZiPS** (from FanGraphs) | Calibration reference, not a component |

### 2.5 Projected lineups

When the actual lineup card is not yet posted (the normal case at prediction time), the
simulator uses a **projected lineup**: each team's modal batting order over its last ~10
games, with players currently on the IL (from `transactions`) removed and replaced by the
next healthy player at that position from the recent depth chart. Stored in `lineups` with
`source = projected`. Actual cards, once posted, are stored alongside with `source =
actual` and used to measure the cost of the projection.

## 3. The fast direct path (feeds the season Monte Carlo)

| Model | Target | Notes |
|---|---|---|
| **Elo** | game win probability | MOV multiplier, starting-pitcher adjustment, home-field constant; strictly chronological, append-only ratings. Also an ensemble input and the "are we adding value?" bar. |
| **Logistic regression** (`direct_v1`) | game win probability | on home−away differenced features; L2-regularised; a transparent, well-calibrated baseline |
| **Poisson / Negative-Binomial GLM** (`direct_v1`) | each team's runs | independent Poisson sides today (μ clamped against divergence); NB for overdispersion and a **bivariate Poisson** coupling are documented refinements |
| **Gradient-boosted trees** (`gbm_v1`) | game win probability | `HistGradientBoostingClassifier` (scikit-learn's histogram GBM, LightGBM family) on the same `fs_v1` diff features; lightly regularised (depth 3, 300 iters, early stopping) because the feature set is small; a non-linear third base model for the stack |

These produce a game win probability and a run distribution in microseconds. The
100k-season Monte Carlo samples game outcomes from them; it does **not** run the PA engine.

## 4. Ensemble & calibration (`mlbsim-models/ensemble/`)

- **Stacking** (`stack.py` → `ensemble_v1`): base win probabilities are mapped to the
  logit scale and a small L2 logistic regression learns per-model weights plus an
  intercept correction. Base models today: `elo_v1`, `direct_v1`, `gbm_v1` (calibrated-sim
  is the next to land). Fit on a time slice; **never** on a base model's own training
  data.
- **Calibration** (`calibrate.py`): `fit_calibrator` picks **Platt scaling** when the
  calibration holdout is under 1,000 points and **isotonic regression** above it —
  isotonic's step function overfits a small holdout badly (it drove `ensemble_v1` *below*
  coin-flip on one season before this guard). Fit on a held-out *later* slice using the
  stack's predictions there. `train_ensemble` reports holdout log loss / Brier / AUC / ECE
  and the ECE improvement over the uncalibrated stack, and records which calibrator was
  chosen.
- **Honest check**: if the calibrated simulator alone matches or beats the stack
  out-of-sample, ship the simulator alone and document it.

## 4a. Explainability (`mlbsim-models/explain/`)

- **Factors** (`factors.py`): for the logistic win model the SHAP value of feature *j* is
  exactly `coef_j · z_j` (standardised features, `E[z_j] = 0`), and
  `logit(p) = intercept + Σ φ_j` — computed directly, no `shap` dependency. `top_factors`
  ranks by absolute effect and reports each as a direction-aware label plus a
  `prob_shift` (how much the home probability drops if that factor is removed). Written to
  `game_predictions.factors.top_factors`. The GBM path will use `shap.TreeExplainer`.
- **Natural language** (`nl_explainer.py`): a deterministic `template` backend (default —
  free, tested) turns the factor list into two or three sentences; an `anthropic` backend
  rephrases the *same* factors via Claude (gated on `ANTHROPIC_API_KEY`, never invents a
  number, not run in CI).

## 4b. LLM prediction experiment

Separate and self-contained — see `docs/LLM_EXPERIMENT.md`. The harness
(`mlbsim_models/llm/`) builds structured game cards, scores whatever probabilities a
zero-shot or LoRA-fine-tuned model returns against `direct_v1` / `ensemble_v1` on identical
splits, and records a plain-English verdict to `serving.llm_experiments`. The comparison
table is published regardless of outcome; "the LLM does not beat the GBM" is an acceptable —
and expected — result.

## 5. Simulation engines

### 5.1 Game simulator (`mlbsim-engine/game.py`)

Plate-appearance model: pitcher → batter (lineup slot) → event-model draw → baserunning
draw → runs / outs → half-inning → bullpen check → game.

**Implementation (M3): vectorised cohort simulation, not a per-game Python loop.** All
`n` simulated games advance together — one PA per iteration for the sims currently
batting — using NumPy array ops over per-sim state vectors (`score`, `inning`, `outs`,
`base`, lineup index, `live`). `n_sims` is processed in cohorts of 25k (M10) so
throughput is flat in `n`. This runs 10k sims/game in ~0.55 s with no compiled
dependency. A Numba/Rust rewrite was evaluated in M10 and declined — the hot path is
NumPy array calls, not a Python loop (see ADR 0002 and `docs/PERFORMANCE.md`).
`test_bench.py` gates the wall-clock and the scaling.

Components:
- **Event rates** — `rates.py`: `odds_ratio_matchup` (log5) combines batter, pitcher and
  league PA-outcome vectors over `{field_out, K, BB, HBP, 1B, 2B, 3B, HR}`.
  `rates_from_stat_json` derives a player's vector from a `player_season_stats` line;
  learned rate models (mlbsim-models `event/`) can replace this later.
- **Baserunning** — `baserunning.py`: a **vectorised rule-based** advancement model with a
  handful of tunable probabilities (`outcomes.py`); handles forces, probabilistic
  first-to-third / second-scores, and a productive-out run from third. An empirical
  transition matrix estimated from 2015+ play-by-play is a documented later refinement.
- **Bullpen** — crude hook: starter rates through inning 6, then a reliever rate vector.

Rules modelled: 2020+ extra-inning ghost runner on 2B, walk-off termination, home team not
batting in the 9th when already ahead. Outputs: win probability, score grid, runs-by-inning
means/distributions, extra-inning probability, per-team shutout probability, one-run-game
probability, most-likely final scores, and per-lineup-slot batting lines
(PA/AB/H/HR/BB/K/TB/RBI). Persisted to `serving.simulation_runs` + `game_sim_results`.

Calibration against the M2 direct models (and adopting the empirical baserunning matrix)
is ongoing. First-cut neutral-game output: ~8.7 combined runs, home win ~.50, P(extra)
~.10, P(one-run) ~.30 — consistent with recent MLB.

### 5.2 Season Monte Carlo (`mlbsim-engine/season.py`, `playoffs.py`)

Vectorised NumPy, **chunked** (5,000 seasons/batch) with online aggregation — the full
seasons × games tensor is never materialised. `simulate_season(SeasonSetup, n_sims)`:

1. Draw every remaining game's winner: `Bernoulli(p_home_win)` where `p_home_win` is
   supplied per game (from `game_predictions` — `direct_v1` / `elo_v1` — with a
   strength-logistic fallback).
2. Accumulate to standings (actual W-L as-of + simulated remainder), grouping the scatter
   by team so it is 30 vectorised adds rather than a giant index array.
3. Seed each league: division winners = best record per division (seeds 1-3 by record),
   wild cards = next 3 (seeds 4-6); seeds 1-2 bye.
4. Run the bracket + World Series via `simulate_series` (best-of-K, loop ≤ 7 masked draws,
   home pattern per round; per-game prob from an Elo-style strength logistic +
   home-field).
5. Tally per team: playoff / division / wild-card / bye / pennant / World-Series
   probability, expected W-L, expected seed, expected postseason game wins, full win
   distribution; per round: P(sweep), expected games, game-count distribution.

**Performance:** 100k seasons ≈ 0.65 s (~167M game-sims/s); 1M ≈ 6.5 s. Probability totals
come out exact by construction (Σ p_playoffs = 12, Σ p_division = 6, Σ p_pennant = 2,
Σ p_WS = 1).

- **Rotation projection.** Probable pitchers are announced only a few days out; beyond that
  the per-game win probability comes from team strength (a rotation-turn projection that
  feeds a starter-aware direct model is a later refinement).
- **Tiebreakers.** The 2022 CBA breaks regular-season ties by formula (head-to-head, then
  intra-division, then last-half record …) — no Game 163. The current sim approximates the
  cascade with a strength-plus-jitter tiebreak; the exact H2H cascade is a documented
  later refinement (it needs per-sim head-to-head tracking).

### 5.3 Playoff bracket format (2022–present, assumed for 2026)

12 teams — per league: 3 division winners + 3 wild cards. Seeds 1–2 get a bye to the
Division Series. Wild Card Series is **best-of-3** (higher seed hosts all games); Division
Series **best-of-5**; Championship Series and World Series **best-of-7**. `series_predictions`
carries `best_of` so game-count distributions are interpreted correctly.

### 5.4 Playoff / series simulator (`mlbsim-engine/playoffs.py`)

From each simulated seeding: simulate Wild Card / Division / Championship / World Series
rounds under the format above. Per-game win probability from the direct path with
**rotation alignment** (game 1 starter, game 2 starter, …) and the correct home-field
pattern per round (2-2-1 for best-of-5, 2-3-2 for best-of-7, all-home for best-of-3).
Outputs per series: P(each team wins), expected games, P(sweep), full game-count
distribution.

### 5.5 Live in-game win probability (`mlbsim-engine/game.py`, `mlbsim-models/pipelines/live.py`)

No separate win-probability model and no historical backfill — the live number is the
game simulator **resumed from the current state**. `GameState` (inning, half, outs, base
mask, score, next batter per side, starter-pulled flags) seeds `simulate_game(start_state=…)`
so all `n` sims begin from that situation and play only the remainder; `live_win_probability()`
returns P(home win) off that cohort (~6,000 sims, well under a tenth of a second).

`state_from_feed()` maps a GUMBO `feed/live` document → `GameState`: `linescore` gives
inning / half / outs / runs and the `offense.first|second|third` occupancy → 3-bit base
mask, `offense.battingOrder` → lineup index for the batting side. A starter is treated as
pulled once the inning ≥ 6 **or** the current pitcher differs from the announced probable;
the bullpen then uses the same starter-rate fallback as the pre-game path (per-reliever
rates are a documented later refinement). `update_live_games(date)` polls the day's
schedule, and for every game whose `abstractGameState` is `Live` writes a
`game_predictions` row with `is_live = true`, the serialised `game_state_json`, and the
`live_v1` model id. The daily flow never resolves `is_live` rows — only the final,
pre-game prediction is scored in `prediction_outcomes`.

## 6. Explainability

- **SHAP** on the GBM win/runs models → top positive/negative factors per game, stored in
  `game_predictions.factors` and surfaced on the game page.
- **NL explainer** (LLM): converts SHAP factors + head-to-head feature deltas into a short
  readable paragraph ("The model favours the Dodgers primarily because of a starting-
  pitcher edge (…) and stronger recent offensive production (…), partly offset by …").
  The LLM only narrates numbers the models produced — it never invents a prediction.

## 7. LLM / fine-tuning experiment (honest, self-contained)

Tracked in `serving.llm_experiments`. Candidate tasks:

1. Fine-tune a small open model (LoRA) on structured game descriptions → predict win
   probability; benchmark against the GBM/ensemble on identical time splits.
2. Player/team **embeddings** from a fine-tuned encoder; test whether they improve the
   event model or projections.
3. LLM-as-reasoner over a structured game card; compare Brier / log loss to the ensemble.

Reporting rule: publish the comparison table regardless of outcome. If the LLM does not
beat conventional models, that is the finding and it is stated without hedging.

## 8. Data-leakage register

Leakage is the primary way this project could be quietly wrong. Explicit mitigations:

| Risk | Mitigation |
|---|---|
| Rolling / season-to-date stats include the game being predicted | All feature builders use the blessed `asof.py` join: `stat.game_date < prediction.as_of_ts` (strict `<`). Season aggregates explicitly exclude the target `game_pk`. Property test: a feature "as of D" never changes when future rows are added. |
| Model artifact overlaps the test period | `model_versions` stores `train_start` / `train_end`; scoring guard refuses games with `game_date <= train_end`. Backtests instantiate the model version that would have existed on that date (walk-forward retrain). |
| Lineups known only ~2–3 h pre-game; probables earlier | `game_probables` and `lineups` carry `source` (projected/actual) + `source_ts`. System backtests use projected lineups + probable pitchers. Actual lineups are training-only for the event model. |
| Park factors / league run environment from the season being predicted | Trailing 3-year park factors; prior-season league wOBA scale. Never current-season realised values. |
| Elo peeking | Single-pass chronological; ratings table append-only keyed by `as_of_date`. |
| Target / mean encoding fit on full data | Encoders fit inside CV folds via sklearn `Pipeline`; never `.fit()` on concatenated train+val. |
| Weather = final observed instead of forecast | Store the forecast pulled at `scheduled_start − 24h`. Observed weather is a separate column used only post-hoc. |
| Injuries / IL status | `transactions` with effective dates; roster availability reconstructed as-of, not from end-of-season rosters. |
| Standings / playoff-race features leak future results | `standings.source` = actual/projected; features join the `actual` snapshot strictly before `as_of`. |
| Season-total xStats used to predict a mid-season game | xwOBA etc. are valid inputs but must be as-of-dated like any other rolling stat. |

Every feature row also stores `data_snapshot_hash`; CI reproduces it to catch accidental
future-data contamination.
