"""Unit test for the upsert helper's SQL shape (no live database)."""

from __future__ import annotations

from sqlalchemy.dialects import postgresql

from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import Player


class _RecordingSession:
    """Minimal stand-in that captures compiled statements."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, stmt) -> None:
        self.statements.append(
            str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
        )


def test_empty_rows_is_noop():
    session = _RecordingSession()
    assert upsert(session, Player, [], index_elements=["player_id"]) == 0
    assert session.statements == []


def test_emits_on_conflict_do_update():
    session = _RecordingSession()
    rows = [{"player_id": 605141, "full_name": "Mookie Betts", "retro_id": "bettm001"}]
    sent = upsert(
        session,
        Player,
        rows,
        index_elements=["player_id"],
        update_columns=["full_name", "retro_id"],
    )
    assert sent == 1
    sql = session.statements[0].lower()
    assert "insert into warehouse.players" in sql
    assert "on conflict (player_id) do update" in sql
    assert "full_name" in sql and "retro_id" in sql


def test_chunks_large_batches():
    session = _RecordingSession()
    rows = [{"player_id": i, "full_name": f"P{i}"} for i in range(2500)]
    sent = upsert(session, Player, rows, index_elements=["player_id"], chunk_size=1000)
    assert sent == 2500
    assert len(session.statements) == 3
