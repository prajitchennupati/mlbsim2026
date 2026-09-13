"""Structured game cards, a zero-shot LLM classifier, and the comparison harness.

The fine-tuned (LoRA) run itself needs GPU compute and is executed out of band;
this module builds the inputs, scores whatever probabilities come back against the
GBM/ensemble on identical splits, and records the verdict in
``serving.llm_experiments``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections.abc import Sequence
from typing import Any

from sqlalchemy import insert

from mlbsim_core import get_logger, session_scope
from mlbsim_data.models import LlmExperiment
from mlbsim_models.evaluate.metrics import brier_score, log_loss, roc_auc

_log = get_logger(__name__)

_SYSTEM = (
    "You are a baseball prediction model. Given a structured pre-game card, output ONLY a "
    'JSON object {"home_win_prob": p} where p is your probability the home team wins, '
    "0 < p < 1. Use the numbers in the card; do not invent information."
)


def game_card(context: dict[str, Any]) -> str:
    """Deterministic structured text for one game — the LLM input / fine-tuning example."""
    fields = [
        ("matchup", f"{context.get('away', 'AWAY')} @ {context.get('home', 'HOME')}"),
        ("date", context.get("date", "")),
        ("home_record", context.get("home_record", "")),
        ("away_record", context.get("away_record", "")),
        ("home_rs_pg", context.get("home_rs_pg")),
        ("home_ra_pg", context.get("home_ra_pg")),
        ("away_rs_pg", context.get("away_rs_pg")),
        ("away_ra_pg", context.get("away_ra_pg")),
        ("home_sp_era", context.get("home_sp_era")),
        ("away_sp_era", context.get("away_sp_era")),
        ("home_last10", context.get("home_last10")),
        ("away_last10", context.get("away_last10")),
        ("park", context.get("park", "")),
    ]
    return "\n".join(f"{k}: {v}" for k, v in fields if v is not None and v != "")


def _parse_prob(text: str) -> float:
    try:
        obj = json.loads(text[text.index("{") : text.rindex("}") + 1])
        p = float(obj["home_win_prob"])
    except (ValueError, KeyError):
        m = re.search(r"0?\.\d+", text)
        if not m:
            raise ValueError(f"no probability in LLM output: {text!r}") from None
        p = float(m.group())
    return min(max(p, 1e-4), 1.0 - 1e-4)


def zero_shot_win_probs(
    cards: Sequence[str], *, model: str = "claude-sonnet-5", max_tokens: int = 40
) -> list[float]:
    """Query the LLM for P(home win) per card. Requires ``ANTHROPIC_API_KEY``."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for zero_shot_win_probs")
    import anthropic

    client = anthropic.Anthropic()
    out: list[float] = []
    for card in cards:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": card}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        out.append(_parse_prob(text))
    return out


def compare_and_verdict(
    llm_probs: Sequence[float],
    baseline_probs: Sequence[float],
    y: Sequence[int],
    *,
    baseline_name: str = "direct_v1",
) -> dict[str, Any]:
    """Score the LLM vs a baseline on the same games and state a plain-English verdict."""
    llm = {
        "log_loss": log_loss(y, llm_probs),
        "brier": brier_score(y, llm_probs),
        "roc_auc": roc_auc(y, llm_probs),
    }
    base = {
        "log_loss": log_loss(y, baseline_probs),
        "brier": brier_score(y, baseline_probs),
        "roc_auc": roc_auc(y, baseline_probs),
    }
    beats = llm["log_loss"] < base["log_loss"]
    verdict = (
        f"On {len(y)} games the LLM "
        + ("BEATS" if beats else "does NOT beat")
        + f" {baseline_name} on log loss "
        f"({llm['log_loss']:.4f} vs {base['log_loss']:.4f}; "
        f"Brier {llm['brier']:.4f} vs {base['brier']:.4f}; "
        f"AUC {llm['roc_auc']:.3f} vs {base['roc_auc']:.3f})."
    )
    return {"llm": llm, "baseline": base, "beats_baseline": beats, "verdict": verdict}


def record_experiment(
    *,
    base_model: str,
    task: str,
    train_spec: dict[str, Any],
    metrics: dict[str, Any],
    verdict: str,
) -> int:
    """Persist one row to ``serving.llm_experiments``. Returns the exp_id."""
    with session_scope() as s:
        exp_id = s.execute(
            insert(LlmExperiment)
            .values(
                base_model=base_model,
                task=task,
                train_spec_json=train_spec,
                metrics_json=metrics,
                verdict=verdict[:2000],
                created_at=dt.datetime.now(dt.UTC),
            )
            .returning(LlmExperiment.exp_id)
        ).scalar_one()
    _log.info("llm.experiment.recorded", exp_id=exp_id, task=task, base_model=base_model)
    return int(exp_id)
