"""Generic PostgreSQL ``INSERT ... ON CONFLICT DO UPDATE`` helper.

Re-running any load is safe: rows are matched on ``index_elements`` (a primary
key or unique constraint) and non-key columns are refreshed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from mlbsim_core.db import Base


def upsert(
    session: Session,
    model: type[Base],
    rows: Sequence[dict[str, Any]],
    *,
    index_elements: Sequence[str],
    update_columns: Sequence[str] | None = None,
    chunk_size: int = 1000,
) -> int:
    """Upsert ``rows`` into ``model``'s table. Returns the number of rows sent.

    ``update_columns`` defaults to every inserted column except the conflict
    key(s), so a repeated load overwrites stale values with fresh ones.
    """
    if not rows:
        return 0

    all_columns = list(rows[0].keys())
    conflict = set(index_elements)
    updatable = [
        c
        for c in (update_columns if update_columns is not None else all_columns)
        if c in all_columns and c not in conflict
    ]

    sent = 0
    for start in range(0, len(rows), chunk_size):
        batch = rows[start : start + chunk_size]
        stmt = insert(model).values(batch)
        if updatable:
            stmt = stmt.on_conflict_do_update(
                index_elements=list(index_elements),
                set_={col: getattr(stmt.excluded, col) for col in updatable},
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=list(index_elements))
        session.execute(stmt)
        sent += len(batch)
    return sent
