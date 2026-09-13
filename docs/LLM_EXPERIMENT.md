# LLM Experiment — does an LLM beat the GBM/ensemble?

**Position:** structured tabular prediction is a solved problem for gradient boosting and
calibrated logistic models. The LLM's honest job in this project is turning SHAP factors
into readable prose (see `mlbsim_models.explain.nl_explainer`). This document defines a
separate, self-contained experiment that tests whether an LLM can *predict* — and commits
to publishing the result either way.

## Tasks under test

| id | description | model | status |
|---|---|---|---|
| `win_prob_zeroshot` | Claude reads a structured game card, returns `{"home_win_prob": p}` | `claude-sonnet-5`, no fine-tuning | harness ready; not yet run (needs API budget) |
| `win_prob_lora` | small open model (Llama-3.1-8B / Mistral-7B) LoRA-fine-tuned on ~30k historical game cards → win probability | `<base>+lora` | harness ready; not yet run (needs GPU) |
| `embeddings` | fine-tuned encoder → team/player vectors; test whether they improve the event model or projections | — | future |

## Protocol (identical to every other model)

- **Input:** the deterministic `game_card(context)` string — matchup, records, RS/RA per
  game, starter ERAs, last-10 form, park. Same point-in-time discipline as the feature
  store: only information available before first pitch.
- **Splits:** train ≤ 2022 · validate 2023 · **test 2024**, then walk-forward through 2025.
  The LoRA model is trained only on cards dated before the test window.
- **Baselines it must beat:** `direct_v1` (logistic + Poisson) and `ensemble_v1` (stacked +
  isotonic), scored on the *same* test games.
- **Metrics:** log loss (primary), Brier, ROC-AUC, ECE + reliability curve.
- **Recording:** `mlbsim_models.llm.experiment.compare_and_verdict(...)` produces a
  plain-English verdict; `record_experiment(...)` writes it to `serving.llm_experiments`
  with the full metrics and the train spec.

## Reporting rule

The comparison table below is published **regardless of outcome**. If the LLM does not beat
`direct_v1` / `ensemble_v1`, that is the finding and it is stated plainly — that outcome is
more valuable to the project's thesis (understanding *why* a model works) than an
artificially favourable one.

## Results

_Pending the runs above. Filled from `serving.llm_experiments` once executed._

| task | model | n (test) | log loss | Brier | AUC | vs `direct_v1` | verdict |
|---|---|---|---|---|---|---|---|
| `win_prob_zeroshot` | claude-sonnet-5 | — | — | — | — | — | _to be run_ |
| `win_prob_lora` | llama-3.1-8b+lora | — | — | — | — | — | _to be run_ |

## Why zero-shot is expected to underperform

A calibrated GBM sees thousands of labelled games and learns the precise weight of a 0.3-run
RS/g edge or a 0.4 ERA starter gap. A zero-shot LLM reasons qualitatively from the same
card and tends to be under-confident (probabilities bunched near 0.5) and mildly
miscalibrated. LoRA fine-tuning closes some of that gap but, on purely tabular signal with
no text to exploit, rarely surpasses gradient boosting. The experiment exists to measure
that gap on *this* data, not to assume it.
