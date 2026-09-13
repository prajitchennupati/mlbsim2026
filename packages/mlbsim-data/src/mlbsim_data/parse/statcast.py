"""Parse the Baseball Savant Statcast CSV into ``batted_balls`` enrichment rows.

Join key: ``pa_id = game_pk * 1000 + (at_bat_number - 1)`` — Savant's 1-indexed
``at_bat_number`` equals the Stats API 0-indexed ``atBatIndex`` plus one.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd

from mlbsim_core import get_logger

_log = get_logger(__name__)

_HARD_HIT_MPH = 95.0
_BARREL_CLASS = 6  # launch_speed_angle classification: 6 == "barrel"

_USECOLS = [
    "game_pk",
    "at_bat_number",
    "type",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "launch_speed_angle",
]


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def parse_statcast_batted_balls(csv_text: str) -> list[dict[str, Any]]:
    """One enrichment row per batted ball (rows where ``type == 'X'``)."""
    frame = pd.read_csv(
        StringIO(csv_text),
        usecols=lambda c: c in _USECOLS,
        dtype=str,
        keep_default_na=False,
    )
    rows: list[dict[str, Any]] = []
    for r in frame.itertuples(index=False):
        if getattr(r, "type", "") != "X":
            continue
        game_pk = _num(getattr(r, "game_pk", None))
        at_bat = _num(getattr(r, "at_bat_number", None))
        if game_pk is None or at_bat is None:
            continue
        launch_speed = _num(getattr(r, "launch_speed", None))
        lsa = _num(getattr(r, "launch_speed_angle", None))
        rows.append(
            {
                "pa_id": int(game_pk) * 1000 + (int(at_bat) - 1),
                "launch_speed": launch_speed,
                "launch_angle": _num(getattr(r, "launch_angle", None)),
                "total_distance": (
                    int(d) if (d := _num(getattr(r, "hit_distance_sc", None))) is not None else None
                ),
                "xba": _num(getattr(r, "estimated_ba_using_speedangle", None)),
                "xwoba": _num(getattr(r, "estimated_woba_using_speedangle", None)),
                "is_barrel": None if lsa is None else int(lsa) == _BARREL_CLASS,
                "is_hardhit": None if launch_speed is None else launch_speed >= _HARD_HIT_MPH,
            }
        )
    _log.info("statcast.parsed_batted_balls", n=len(rows))
    return rows
