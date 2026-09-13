"""Turn a prediction + its factors into a readable paragraph.

The default ``template`` backend is deterministic, free, and fully tested. The
``anthropic`` backend only rephrases the same factor list — it never invents a
number — and requires ``ANTHROPIC_API_KEY``; it is exercised manually, not in CI.
"""

from __future__ import annotations

import os
from typing import Any

_PROMPT_SYSTEM = (
    "You explain a baseball model's game prediction to a knowledgeable fan in two or "
    "three sentences. Use ONLY the numbers provided. Do not invent statistics, players, "
    "or injuries. Lead with the win probability, then the two or three biggest factors."
)


def _pct(p: float) -> str:
    return f"{round(100 * p)}%"


def build_prompt(context: dict[str, Any], factors: list[dict[str, Any]]) -> dict[str, str]:
    """Assemble the system + user prompt for the LLM backend."""
    home, away = context.get("home", "the home team"), context.get("away", "the away team")
    lines = [
        f"Matchup: {away} at {home}",
        f"Model win probability: {home} {_pct(context['home_win_prob'])}, "
        f"{away} {_pct(1 - context['home_win_prob'])}",
    ]
    if context.get("exp_home_runs") is not None:
        lines.append(
            f"Expected score: {home} {context['exp_home_runs']:.1f}, "
            f"{away} {context['exp_away_runs']:.1f}"
        )
    lines.append("Top factors (SHAP, logit scale; positive favours the home team):")
    for f in factors:
        side = home if f["favours"] == "home" else away
        lines.append(f"  - {f['label']}: favours {side}, probability shift {f['prob_shift']:+.3f}")
    return {"system": _PROMPT_SYSTEM, "user": "\n".join(lines)}


def _template_explanation(context: dict[str, Any], factors: list[dict[str, Any]]) -> str:
    home, away = context.get("home", "the home team"), context.get("away", "the away team")
    p = context["home_win_prob"]
    favourite, prob = (home, p) if p >= 0.5 else (away, 1 - p)
    parts = [f"The model gives {favourite} a {_pct(prob)} chance to win"]
    if context.get("exp_home_runs") is not None:
        parts.append(
            f", with an expected score of {context['exp_home_runs']:.1f}-"
            f"{context['exp_away_runs']:.1f}"
        )
    parts.append(". ")

    for_fav = [f for f in factors if (f["favours"] == "home") == (favourite == home)]
    against = [f for f in factors if f not in for_fav]
    if for_fav:
        reasons = ", ".join(f["label"] for f in for_fav[:3])
        parts.append(f"The main reasons are {reasons}")
    if against:
        parts.append(f", partly offset by {against[0]['label']}")
    parts.append(".")
    return "".join(parts)


def render_explanation(
    context: dict[str, Any],
    factors: list[dict[str, Any]],
    *,
    backend: str = "template",
    model: str = "claude-sonnet-5",
) -> str:
    """Return a short natural-language explanation of the prediction."""
    if backend == "template":
        return _template_explanation(context, factors)
    if backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for the 'anthropic' backend")
        import anthropic

        prompt = build_prompt(context, factors)
        msg = anthropic.Anthropic().messages.create(
            model=model,
            max_tokens=200,
            system=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
        )
        return "".join(
            block.text for block in msg.content if getattr(block, "type", "") == "text"
        ).strip()
    raise ValueError(f"unknown backend {backend!r}")
