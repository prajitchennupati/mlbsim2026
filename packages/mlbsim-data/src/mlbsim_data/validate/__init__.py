"""Dataframe-level validation of parsed rows before they hit the warehouse.

A failing batch raises ``pandera.errors.SchemaError`` and is not loaded — bad
data quarantines rather than partially loading.
"""

from mlbsim_data.validate.schemas import (
    BATTING_GAME_LOG_SCHEMA,
    GAME_SCHEMA,
    PLATE_APPEARANCE_SCHEMA,
    validate_rows,
)

__all__ = [
    "BATTING_GAME_LOG_SCHEMA",
    "GAME_SCHEMA",
    "PLATE_APPEARANCE_SCHEMA",
    "validate_rows",
]
