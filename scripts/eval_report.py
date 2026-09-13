"""Walk-forward evaluation report for the win-probability models.

Assumes the warehouse already has a contiguous block of finished games ingested.
Trains through ``--cutoff``, predicts every game after it, and prints a Markdown
table of each model vs the coin-flip and home-team baselines. Paste the output
into ``docs/EVALUATION.md`` §6.

    python scripts/eval_report.py --cutoff 2024-07-01
"""

from __future__ import annotations

import argparse

from sqlalchemy import select

from mlbsim_core import session_scope
from mlbsim_data.models import Game
from mlbsim_features.pipeline import build_game_features
from mlbsim_models.pipelines import (
    build_elo,
    evaluate_model,
    predict_date,
    predict_ensemble_date,
    predict_gbm_date,
    train_direct_models,
    train_ensemble,
    train_gbm,
)
from mlbsim_models.pipelines.resolve import resolve_outcomes


def _seasons_and_dates(cutoff: str) -> tuple[list[int], list[str], list[str]]:
    """Scope the whole report to the cutoff's season."""
    season = int(cutoff[:4])
    with session_scope() as s:
        all_dates = [
            r[0].isoformat()
            for r in s.execute(
                select(Game.game_date)
                .where(Game.season == season, Game.home_score.is_not(None), Game.game_type == "R")
                .distinct()
                .order_by(Game.game_date)
            )
        ]
    test_dates = [d for d in all_dates if d >= cutoff]
    return [season], all_dates, test_dates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True, help="train on games strictly before this date")
    args = ap.parse_args()

    seasons, all_dates, test_dates = _seasons_and_dates(args.cutoff)
    print(f"seasons in warehouse: {seasons}")
    print(
        f"train dates: {all_dates[0]} .. < {args.cutoff}  "
        f"({len(all_dates) - len(test_dates)})   "
        f"test dates: {test_dates[0]} .. {test_dates[-1]}  ({len(test_dates)})\n",
        flush=True,
    )

    print("building features + elo ...", flush=True)
    build_game_features(seasons=seasons)
    build_elo(seasons=seasons)

    print(f"training direct + gbm through {args.cutoff}, predicting every date ...", flush=True)
    train_direct_models(train_end=args.cutoff)
    train_gbm(train_end=args.cutoff)
    for d in all_dates:  # train-window base preds feed the stacker; test-window are OOS
        predict_date(d)
        predict_gbm_date(d)

    print(f"training ensemble through {args.cutoff}, predicting the test window ...", flush=True)
    train_ensemble(train_end=args.cutoff)
    for d in test_dates:
        predict_ensemble_date(d)

    resolve_outcomes(since=f"{seasons[0]}-01-01")

    rows = []
    for mid in ("elo_v1", "direct_v1", "gbm_v1", "ensemble_v1"):
        try:
            m = evaluate_model(
                mid,
                start=args.cutoff,
                end=test_dates[-1],
                persist=False,
                split_name=f"walkforward_{args.cutoff}",
            )
        except LookupError as exc:
            print(f"  {mid}: {exc}")
            continue
        r = m["model"]
        cf, ht = m["baselines"]["coin_flip"], m["baselines"]["home_team"]
        rows.append(
            (
                mid,
                r["n"],
                r["accuracy"],
                r["log_loss"],
                r["brier"],
                r["roc_auc"],
                r["ece"],
                cf["log_loss"],
                ht["log_loss"],
                m["beats_coin_flip_logloss"],
                m["beats_home_team_logloss"],
            )
        )

    print(
        "\n| model | n | acc | log loss | Brier | AUC | ECE | vs coin-flip LL | "
        "vs home-team LL | beats CF | beats HT |"
    )
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|")
    for mid, n, acc, ll, br, auc, ece, cfll, htll, bcf, bht in rows:
        print(
            f"| {mid} | {int(n)} | {acc:.3f} | {ll:.4f} | {br:.4f} | {auc:.3f} | {ece:.3f} "
            f"| {cfll:.4f} | {htll:.4f} | {'yes' if bcf else 'no'} | {'yes' if bht else 'no'} |"
        )
    print()


if __name__ == "__main__":
    main()
