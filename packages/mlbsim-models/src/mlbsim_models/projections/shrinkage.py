"""Empirical-Bayes partial pooling for player rates.

For each outcome, shrink a player's observed rate toward the population mean by
``n / (n + k)`` where ``k`` is a beta-binomial method-of-moments estimate from
the population's mean and variance. This is the lightweight "hierarchical"
upgrade over Marcel; a full PyMC partial-pooling model is a documented later
experiment (``docs/MODELING.md`` §2.4).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
_EPS = 1e-9


def beta_binomial_k(pop_mean: Array, pop_var: Array) -> Array:
    """Method-of-moments strength for a beta prior; larger k = more shrinkage."""
    denom = np.maximum(pop_var, _EPS)
    k = pop_mean * (1.0 - pop_mean) / denom - 1.0
    return np.clip(k, 1.0, 1e6)


def shrink_rates(
    player_rate: Array,
    player_n: float,
    population: Array,
) -> Array:
    """Shrink one player's 8-outcome rate vector toward the population.

    ``population`` is an ``(m, 8)`` matrix of qualified players' rate vectors used
    to estimate the prior. Returns a re-normalised 8-vector.
    """
    pop = np.asarray(population, dtype=np.float64)
    pop_mean = pop.mean(axis=0)
    pop_var = pop.var(axis=0)
    k = beta_binomial_k(pop_mean, pop_var)
    w = player_n / (player_n + k)
    shrunk = w * np.asarray(player_rate, dtype=np.float64) + (1.0 - w) * pop_mean
    total = shrunk.sum()
    return np.asarray(shrunk / total if total else pop_mean, dtype=np.float64)
