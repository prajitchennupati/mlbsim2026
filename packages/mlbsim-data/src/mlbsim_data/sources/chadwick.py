"""Chadwick Bureau register — the player-ID crosswalk.

``people.csv`` maps a person across ID systems (MLBAM, Retrosheet,
Baseball-Reference, FanGraphs). We keep rows that (a) carry an MLBAM id and
(b) actually appeared in an MLB game.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from io import StringIO

import pandas as pd

from mlbsim_core import get_logger, get_settings
from mlbsim_data.sources.http import fetch_text

_log = get_logger(__name__)

_USECOLS = [
    "key_mlbam",
    "key_retro",
    "key_bbref",
    "key_fangraphs",
    "name_first",
    "name_last",
    "birth_year",
    "birth_month",
    "birth_day",
    "mlb_played_first",
    "mlb_played_last",
]


@dataclass(frozen=True, slots=True)
class PlayerXRef:
    """One reconciled player identity."""

    player_id: int  # MLBAM
    retro_id: str | None
    bbref_id: str | None
    fg_id: str | None
    full_name: str
    birth_date: dt.date | None
    mlb_debut_year: int | None
    mlb_final_year: int | None


def _to_int(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return None


def _to_date(y: object, m: object, d: object) -> dt.date | None:
    year, month, day = _to_int(y), _to_int(m), _to_int(d)
    if year is None or month is None or day is None:
        return None
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def _clean_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def parse_people(csv_text: str) -> list[PlayerXRef]:
    """Parse ``people.csv`` text into :class:`PlayerXRef` records (MLB players only)."""
    frame = pd.read_csv(
        StringIO(csv_text),
        usecols=lambda c: c in _USECOLS,
        dtype=str,
        keep_default_na=True,
    )
    records: list[PlayerXRef] = []
    for row in frame.itertuples(index=False):
        raw_mlbam = _clean_str(getattr(row, "key_mlbam", None))
        played_last = _clean_str(getattr(row, "mlb_played_last", None))
        if raw_mlbam is None or played_last is None:
            continue
        first = _clean_str(getattr(row, "name_first", None)) or ""
        last = _clean_str(getattr(row, "name_last", None)) or ""
        played_first = _clean_str(getattr(row, "mlb_played_first", None))
        records.append(
            PlayerXRef(
                player_id=int(float(raw_mlbam)),
                retro_id=_clean_str(getattr(row, "key_retro", None)),
                bbref_id=_clean_str(getattr(row, "key_bbref", None)),
                fg_id=_clean_str(getattr(row, "key_fangraphs", None)),
                full_name=f"{first} {last}".strip(),
                birth_date=_to_date(
                    getattr(row, "birth_year", None),
                    getattr(row, "birth_month", None),
                    getattr(row, "birth_day", None),
                ),
                mlb_debut_year=int(float(played_first)) if played_first else None,
                mlb_final_year=int(float(played_last)),
            )
        )
    _log.info("chadwick.parsed", n_records=len(records))
    return records


_ONE_DAY = 24 * 3600


def fetch_people(*, use_cache: bool = True) -> list[PlayerXRef]:
    """Download (or read from the landing zone) and parse the Chadwick register."""
    settings = get_settings()
    text = fetch_text(
        settings.chadwick_register_url,
        namespace="chadwick",
        key="people",
        suffix=".csv",
        max_age_seconds=_ONE_DAY if use_cache else 0,
    )
    return parse_people(text)
