"""The one blessed point-in-time join.

Rule: a feature computed *as of* timestamp ``t`` may read a source row only if
that row's information was available strictly before ``t``. Every feature builder
in this package routes its historical lookups through :func:`asof_lookup` /
:func:`asof_join`, which use a strict ``<`` bound (never ``<=``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

Row = dict[str, Any]


def _as_date(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def before(rows: Iterable[Row], as_of: dt.date | str, *, date_key: str = "game_date") -> list[Row]:
    """Return ``rows`` whose ``date_key`` is strictly before ``as_of``, oldest first."""
    cutoff = _as_date(as_of)
    kept = [r for r in rows if _as_date(r[date_key]) < cutoff]
    kept.sort(key=lambda r: _as_date(r[date_key]))
    return kept


def asof_lookup(
    rows: Iterable[Row],
    as_of: dt.date | str,
    *,
    date_key: str = "game_date",
) -> Row | None:
    """The most recent row strictly before ``as_of`` (or ``None``)."""
    kept = before(rows, as_of, date_key=date_key)
    return kept[-1] if kept else None


def asof_join(
    events: Sequence[Row],
    queries: Sequence[Row],
    *,
    by: Sequence[str],
    event_date_key: str = "game_date",
    query_date_key: str = "as_of",
    suffix: str = "_asof",
) -> list[Row]:
    """Attach, to each query row, the latest event row (matching ``by``) strictly
    before the query's ``as_of`` date. Non-matching queries get ``None`` fields.
    """
    buckets: dict[tuple[Any, ...], list[Row]] = {}
    for e in events:
        key = tuple(e[k] for k in by)
        buckets.setdefault(key, []).append(e)

    out: list[Row] = []
    for q in queries:
        key = tuple(q[k] for k in by)
        match = asof_lookup(buckets.get(key, []), q[query_date_key], date_key=event_date_key)
        merged = dict(q)
        for k, v in (match or {}).items():
            if k in by or k == event_date_key:
                continue
            merged[f"{k}{suffix}"] = v
        out.append(merged)
    return out


def snapshot_hash(source_ids: Iterable[Any], *, max_source_ts: Any = None) -> str:
    """Deterministic hash of the source rows that fed a feature computation.

    Two runs of a builder for the same ``(entity, feature_set, as_of)`` must
    produce the same hash; CI asserts this to catch accidental future-data leaks.
    """
    payload = {
        "ids": sorted(str(i) for i in source_ids),
        "max_ts": None if max_source_ts is None else str(max_source_ts),
    }
    blob = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
