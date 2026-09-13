"""The live flow: refresh in-game win probabilities for every in-progress game."""

from __future__ import annotations

import datetime as dt
from typing import Any

from mlbsim_core import get_logger
from mlbsim_pipeline.flows._run import FlowRun, ping_revalidate

_log = get_logger(__name__)


def run_live(
    date: str | None = None,
    *,
    n_sims: int = 6_000,
    publish: bool = True,
) -> dict[str, Any]:
    """Poll the schedule for live games and write ``is_live`` win-prob predictions."""
    day = date or dt.date.today().isoformat()
    run = FlowRun(flow="live")

    summary = run.step("update_live_games", lambda: _update(day, n_sims))
    wrote = bool(summary and getattr(summary, "written", 0))
    if publish and wrote:
        run.step("revalidate", ping_revalidate)

    out = run.summary()
    out["date"] = day
    _log.info("flow.live.done", **{k: out[k] for k in ("ok", "n_failed", "elapsed_ms")})
    return out


def _update(day: str, n_sims: int) -> Any:
    from mlbsim_models.pipelines import update_live_games

    return update_live_games(day, n_sims=n_sims)
