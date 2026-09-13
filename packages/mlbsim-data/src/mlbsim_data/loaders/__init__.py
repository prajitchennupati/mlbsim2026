"""Idempotent upserts into the ``warehouse`` schema."""

from mlbsim_data.loaders import warehouse
from mlbsim_data.loaders.upsert import upsert

__all__ = ["upsert", "warehouse"]
