"""Marcel projections (Tom Tango).

Given up to three prior seasons, weight them 5 / 4 / 3, regress the blended rate
toward league average by a fixed "regression PA" amount, and apply an age
adjustment centred on 29. Output is a PA-outcome rate vector (engine order) plus
a projected plate-appearance count, so it drops straight into the simulator.

Simplifications vs the canonical Marcel: a single regression amount for the whole
rate vector (rather than per-component), and the age curve applied to the
"positive" outcomes only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mlbsim_engine.outcomes import BB, HR, LEAGUE_RATES, S1, S2, S3, K
from mlbsim_engine.rates import rates_from_stat_json

Array = NDArray[np.float64]

_WEIGHTS = (5.0, 4.0, 3.0)  # most-recent year first
_REG_PA_BAT = 1200.0
_REG_PA_PITCH = 750.0
_PEAK_AGE = 29
_AGE_GAIN_YOUNG = 0.006
_AGE_LOSS_OLD = 0.003
_AGE_CLAMP = (0.90, 1.10)
_POSITIVE = (S1, S2, S3, HR, BB)


@dataclass(slots=True)
class MarcelProjection:
    rates: Array  # length-8 PA-outcome vector, engine order
    proj_pa: float
    age_factor: float
    weighted_sample: float
    components: dict[str, float] = field(default_factory=dict)


def age_factor(age: float | None) -> float:
    if age is None:
        return 1.0
    slope = _AGE_GAIN_YOUNG if age <= _PEAK_AGE else _AGE_LOSS_OLD
    return float(np.clip(1.0 + slope * (_PEAK_AGE - age), *_AGE_CLAMP))


def _season_amount(stat: dict[str, Any], *, is_pitcher: bool) -> float:
    raw = stat.get("bf") if is_pitcher else stat.get("pa")
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def project_marcel(
    prior_seasons: list[dict[str, Any] | None],
    *,
    age: float | None = None,
    is_pitcher: bool = False,
) -> MarcelProjection:
    """``prior_seasons`` is [year-1, year-2, year-3] season-stat blobs (or None)."""
    reg_pa = _REG_PA_PITCH if is_pitcher else _REG_PA_BAT
    num = np.zeros(8, dtype=np.float64)
    den = 0.0
    pa_by_year: list[float] = []
    for stat, w in zip(prior_seasons, _WEIGHTS, strict=False):
        if not stat:
            pa_by_year.append(0.0)
            continue
        amount = _season_amount(stat, is_pitcher=is_pitcher)
        pa_by_year.append(amount)
        if amount <= 0:
            continue
        vec = rates_from_stat_json(stat, is_pitcher=is_pitcher)
        num += w * amount * vec
        den += w * amount

    blended = (num + reg_pa * LEAGUE_RATES) / (den + reg_pa)

    af = age_factor(age)
    adj = blended.copy()
    adj[list(_POSITIVE)] *= af
    adj[K] *= 1.0 / np.sqrt(af)  # strikeouts drift the other way with age
    adj[0] = max(1.0 - adj[1:].sum(), 0.02)
    adj = adj / adj.sum()

    pa1 = pa_by_year[0] if len(pa_by_year) > 0 else 0.0
    pa2 = pa_by_year[1] if len(pa_by_year) > 1 else 0.0
    proj_pa = 200.0 + 0.5 * pa1 + 0.1 * pa2

    return MarcelProjection(
        rates=adj,
        proj_pa=round(proj_pa, 1),
        age_factor=round(af, 4),
        weighted_sample=round(den, 1),
        components={
            name: round(float(adj[i]), 5)
            for name, i in (
                ("field_out", 0),
                ("k", K),
                ("bb", BB),
                ("1b", S1),
                ("2b", S2),
                ("3b", S3),
                ("hr", HR),
            )
        },
    )
