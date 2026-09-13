# Evaluation

## 1. Validation protocol

**No random splits. Time-ordered only.**

- **Primary splits:** train ≤ 2022 · validate 2023 · test 2024.
- **Walk-forward:** through 2025, retrain monthly on all data strictly before month *m*,
  predict month *m*, concatenate. This is the realistic estimate of 2026 performance.
- **Rolling / expanding windows:** the event model and projections use an expanding
  window; Elo is inherently sequential.
- **2026:** prospective evaluation only — every prediction logged to
  `serving.model_eval_runs` and reconciled against actuals the next day. The daily flow
  runs `resolve_outcomes()` (`mlbsim-models/pipelines/resolve.py`), which fills
  `serving.prediction_outcomes` with per-prediction Brier / log-loss / hit for each
  finalised **non-live** game (idempotent — a prediction is scored once); `rollup_eval()`
  aggregates the resolved rows into a `model_eval_runs` + `calibration_bins` snapshot per
  model. `is_live` win-probability rows (`live_v1`) are never scored here.

Leakage controls are enforced by construction (see `docs/MODELING.md` §8): the backtest
harness instantiates, for each date, the model version that would have existed then, and
all features come through the as-of join.

## 2. Metrics

### Game win/loss

| Metric | Purpose |
|---|---|
| Accuracy | headline, but least informative |
| Log loss | proper scoring rule, penalises confident errors |
| Brier score | proper scoring rule, decomposable into calibration + refinement |
| ROC-AUC | pure discrimination |
| Expected Calibration Error (ECE) + reliability diagram | are the probabilities honest |

### Score / run predictions

| Metric | Purpose |
|---|---|
| MAE, RMSE | point-estimate error on team runs and total runs |
| Poisson deviance | count-model fit |
| Negative log likelihood | distributional fit of the predicted score pmf |
| PIT histogram | is the full predictive distribution calibrated (flat = good) |

### Player props

- Per-stat MAE / RMSE for PA, H, HR, R, RBI, TB, BB, K vs **Marcel** and vs **Steamer**.
- Log loss + calibration for each binary prop: P(1+ H), P(2+ H), P(HR), P(RBI), P(R),
  P(BB), P(K); pitcher P(5+/6+/7+/8+/10+ K), P(5+/6+/7+ IP), P(quality start).

### Season / playoff simulation

- **Interval coverage:** did realised 2024 final standings land inside the predicted 80 %
  and 95 % win bands? Target coverage ≈ nominal.
- **Playoff-berth Brier:** on berth predictions made at fixed dates (May 1, Jul 1,
  Sep 1), scored against actual berths.
- **Series model:** log loss of pre-series win probabilities vs actual series outcomes
  (backtested on historical postseasons).

## 3. Baselines

Every model is reported next to:

| Baseline | Definition |
|---|---|
| Coin flip | 0.5 every game |
| Home team | always pick home; P = league home win rate |
| **Elo** | the bar for W/L — a model that doesn't beat Elo is reported as not beating it |
| Poisson GLM | the bar for run/score predictions |
| **Marcel** | the bar for player props |
| Market (optional, off by default) | closing moneyline / total, used only as an external reference where licensing permits; never a model input |

## 4. Model performance dashboard (`/model`)

Backed by `model_eval_runs` + `calibration_bins`. Shows, per model version:

- headline metrics vs each baseline, on the fixed test season and on the live 2026 window;
- reliability diagram (calibration curve) with ECE;
- accuracy / log loss over time (rolling weekly);
- predicted-vs-actual scatter for scores;
- season-sim interval coverage;
- the LLM-vs-conventional comparison table.

## 5. Reproducibility

- Every `model_version` row: `git_sha`, `train_start`, `train_end`, `feature_set_id`,
  `hyperparams_json`, `artifact_uri`, `metrics_json`.
- Training data snapshots are content-addressed Parquet; the hash is recorded with the
  model.
- `make eval SEASON=2024 MODEL=<id>` reproduces a published metrics table from scratch.

## 6. Results — walk-forward on 2024 (real)

_Reproduce: `python scripts/eval_report.py --cutoff 2024-06-01` against a warehouse with
the 2024 regular season ingested (2,428 games, 182k plate appearances). Run 2026-08-31._

**Setup.** Features (`fs_v1`) and Elo built over the whole season. Direct models
(`WinLogit` + `RunsPoisson`) trained on games **before 2024-06-01** (~800 games); the
stacked + isotonic-calibrated ensemble stacked on the training window's base predictions.
Every game from **2024-06-01 onward** (n = 1,568 with a resolved outcome) predicted and
scored against the final result. Baselines: coin-flip (0.5) and always-home
(P = league home-win rate ≈ 0.53).

| model | n | acc | log loss | Brier | AUC | ECE | coin-flip LL | home-team LL | beats CF | beats HT |
|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| **elo_v1** | 1568 | 0.557 | **0.6830** | **0.2450** | 0.573 | 0.012 | 0.6931 | 0.6927 | ✅ | ✅ |
| direct_v1 | 1568 | 0.548 | 0.6888 | 0.2478 | 0.549 | 0.010 | 0.6931 | 0.6927 | ✅ | ✅ |
| ensemble_v1 | 1568 | 0.556 | 0.6876 | 0.2471 | 0.570 | 0.032 | 0.6931 | 0.6927 | ✅ | ✅ |

**What this says, honestly.**

- **Elo is the best win model at this data volume.** A small but real edge over both
  baselines (log-loss 0.683 vs 0.693, Brier 0.245, AUC 0.573) and the best calibration
  (ECE 0.012). Being sequential it needs no train/test split and does not overfit.
- **`direct_v1` barely separates from chance out of sample** (log-loss 0.689, AUC 0.549).
  A 13-feature logistic on ~800 training games has thin signal; it does not yet beat Elo,
  and saying so is the point.
- **`ensemble_v1` now beats both baselines but not Elo.** Its first run on this data was
  *worse than coin-flip* (log-loss 1.00) — isotonic regression overfit a ~240-point
  calibration holdout, mapping some probabilities near 0/1 and blowing up the log-loss.
  The fix (`fit_calibrator`: Platt scaling below 1,000 calibration points, isotonic above)
  brought it to 0.688 / ECE 0.032. At one season of history the stack still adds nothing
  over Elo, so **a deployment right now should serve Elo** for win probability. The
  ensemble is expected to earn its place once the 2015+ backfill gives the stacker
  multi-season, monthly-retrained base predictions and a large-enough calibration set for
  isotonic.

This is the reporting rule from §3 in action: the simpler model wins, and that is stated
without hedging.

**`gbm_v1` (added M10 follow-up).** A `HistGradientBoostingClassifier` on the same
`fs_v1` diff features is now the third stack base model. In-sample on the 2026 training
window (~800 games, schedule-only features) it fit noticeably tighter than the logistic
(win AUC 0.66 vs 0.57, log loss 0.666 vs 0.684) — expected from a more flexible model.
Its **out-of-sample** walk-forward line and a refreshed four-model table are a
`python scripts/eval_report.py --cutoff <date>` re-run away; the run was not completed
here because the shared Postgres connection degraded after a long session of bulk
backfills. On the evidence so far the GBM adds capacity but, like `direct_v1`, is
constrained by 13 features and one season — Elo remains the model to beat.

### Score / run models, player props, season sim

Deferred until the multi-season backfill (2015+, per ADR 0001) lands — one season is too
short for the Poisson-deviance / PIT-histogram tables and for season-sim interval coverage
to mean anything. The metric definitions (§2) and the runners (`mlbsim evaluate`,
`resolve_outcomes` / `rollup_eval`) are in place and exercised by the integration suite.
