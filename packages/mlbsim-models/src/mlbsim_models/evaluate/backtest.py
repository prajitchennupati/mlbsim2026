"""Time-ordered validation splits — never a random shuffle.

``walk_forward_splits`` yields expanding-window train / test folds: train on
everything strictly before a cutoff, test on the next block, advance the cutoff.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    fold: int
    train_end: dt.date  # exclusive upper bound for training
    test_start: dt.date
    test_end: dt.date  # inclusive


def month_starts(start: dt.date, end: dt.date) -> list[dt.date]:
    out: list[dt.date] = []
    y, m = start.year, start.month
    while dt.date(y, m, 1) <= end:
        out.append(dt.date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def walk_forward_splits(
    *, first_test_month: dt.date, last_test_month: dt.date
) -> Iterator[WalkForwardSplit]:
    """Yield one fold per calendar month from ``first_test_month`` to ``last_test_month``."""
    for fold, start in enumerate(month_starts(first_test_month.replace(day=1), last_test_month)):
        nxt = (
            dt.date(start.year + 1, 1, 1)
            if start.month == 12
            else dt.date(start.year, start.month + 1, 1)
        )
        yield WalkForwardSplit(
            fold=fold,
            train_end=start,
            test_start=start,
            test_end=nxt - dt.timedelta(days=1),
        )


def split_rows_by_date(
    rows: Sequence[dict[str, Any]],
    split: WalkForwardSplit,
    *,
    date_key: str = "game_date",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition ``rows`` into (train, test) for one fold using strict date bounds."""
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for r in rows:
        d = r[date_key]
        d = d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d)[:10])
        if d < split.train_end:
            train.append(r)
        elif split.test_start <= d <= split.test_end:
            test.append(r)
    return train, test
