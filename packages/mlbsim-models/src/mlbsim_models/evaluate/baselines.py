"""Reference predictors every model is scored against.

Each returns the predicted probability that the **home** team wins, aligned to
an input sequence of game dicts (``home_score`` / ``away_score`` only needed for
labels, not for these predictions).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mlbsim_models.ratings.elo import EloConfig, run_elo


def coin_flip_prob(games: Sequence[dict[str, Any]]) -> list[float]:
    return [0.5] * len(games)


def home_team_prob(games: Sequence[dict[str, Any]], *, home_win_rate: float = 0.54) -> list[float]:
    """Constant league home-field win rate for every game."""
    return [home_win_rate] * len(games)


def home_labels(games: Sequence[dict[str, Any]]) -> list[int]:
    """1 if the home team won, else 0 (games without a final score are dropped upstream)."""
    return [1 if int(g["home_score"]) > int(g["away_score"]) else 0 for g in games]


def elo_probs(games: Sequence[dict[str, Any]], cfg: EloConfig | None = None) -> dict[int, float]:
    """Pre-game home win probability keyed by ``game_pk`` from a chronological Elo run."""
    return {r.game_pk: r.home_win_prob for r in run_elo(games, cfg)}
