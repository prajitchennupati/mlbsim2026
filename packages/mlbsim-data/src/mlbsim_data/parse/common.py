"""Small shared helpers for the Stats API parsers."""

from __future__ import annotations

import datetime as dt
from typing import Any

# 3-bit base-occupancy mask: 1B=1, 2B=2, 3B=4.
BASE_FIRST, BASE_SECOND, BASE_THIRD = 1, 2, 4


def league_code(name: str | None) -> str | None:
    """ "American League" -> "AL"; "National League" -> "NL" (fits teams.league varchar(2))."""
    if not name:
        return None
    return "AL" if name.startswith("American") else "NL" if name.startswith("National") else None


def division_code(name: str | None) -> str | None:
    """ "... East/Central/West" -> "E"/"C"/"W" (fits teams.division varchar(4))."""
    if not name:
        return None
    return {"East": "E", "Central": "C", "West": "W"}.get(name.rsplit(" ", 1)[-1])


def base_mask(*, on_first: Any, on_second: Any, on_third: Any) -> int:
    mask = 0
    if on_first:
        mask |= BASE_FIRST
    if on_second:
        mask |= BASE_SECOND
    if on_third:
        mask |= BASE_THIRD
    return mask


def to_int(value: Any) -> int | None:
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return None


def to_float(value: Any) -> float | None:
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return None


def to_date(value: Any) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def to_datetime(value: Any) -> dt.datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def innings_to_outs(ip: Any) -> int:
    """Convert an 'innings pitched' string like ``6.1`` (6⅓) to whole outs."""
    if ip in (None, ""):
        return 0
    whole, _, frac = str(ip).partition(".")
    return (to_int(whole) or 0) * 3 + {"": 0, "0": 0, "1": 1, "2": 2}.get(frac, 0)


def parse_wind(text: Any) -> tuple[int | None, str | None]:
    """Parse ``'8 mph, Out To CF'`` -> (8, 'Out To CF')."""
    if not text:
        return None, None
    parts = str(text).split(",", 1)
    speed = to_int(parts[0].replace("mph", "").strip())
    direction = parts[1].strip() if len(parts) > 1 else None
    return speed, direction
