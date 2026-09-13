"""LLM experiments — the honest "does a fine-tuned model beat the GBM?" test.

Everything here that touches an API is gated on ``ANTHROPIC_API_KEY`` and is not
run in CI. The evaluation + recording harness (``compare_and_verdict``,
``record_experiment``) is pure and tested.
"""

from mlbsim_models.llm.experiment import (
    compare_and_verdict,
    game_card,
    record_experiment,
    zero_shot_win_probs,
)

__all__ = [
    "compare_and_verdict",
    "game_card",
    "record_experiment",
    "zero_shot_win_probs",
]
