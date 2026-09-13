"""Orchestration flows: ordered, individually-guarded pipeline runs.

Each flow returns a plain-dict summary (JSON-serialisable) describing every step
it attempted, whether it succeeded, and a one-line detail. A failing step is
logged and recorded but does not abort the remaining steps -- the daily run
should still publish whatever it managed to compute.
"""

from mlbsim_pipeline.flows.daily import run_daily
from mlbsim_pipeline.flows.live import run_live
from mlbsim_pipeline.flows.status import pipeline_status

__all__ = ["pipeline_status", "run_daily", "run_live"]
