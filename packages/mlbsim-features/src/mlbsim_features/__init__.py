"""Point-in-time feature engineering.

Contract: a feature computed *as of* timestamp ``t`` may read a source row only
if its information was available strictly before ``t``. Every builder routes
historical lookups through :mod:`mlbsim_features.asof` (strict ``<`` bound).
See ``docs/MODELING.md`` section 8.
"""

from mlbsim_features.asof import asof_join, asof_lookup, before, snapshot_hash
from mlbsim_features.build import (
    DIFF_FEATURE_ORDER,
    FEATURE_SET_ID,
    assemble_game_features,
    feature_vector,
    pitcher_form,
    team_form,
)

__all__ = [
    "DIFF_FEATURE_ORDER",
    "FEATURE_SET_ID",
    "asof_join",
    "asof_lookup",
    "assemble_game_features",
    "before",
    "feature_vector",
    "pitcher_form",
    "snapshot_hash",
    "team_form",
]

__version__ = "0.0.0"
