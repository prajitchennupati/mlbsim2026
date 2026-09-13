from __future__ import annotations

from mlbsim_core.db import Base
from mlbsim_data import models  # noqa: F401  (import registers tables on Base.metadata)


def test_dimension_tables_registered_in_warehouse_schema():
    tables = Base.metadata.tables
    for expected in (
        "warehouse.parks",
        "warehouse.teams",
        "warehouse.players",
        "warehouse.seasons",
    ):
        assert expected in tables, f"missing table {expected}"


def test_players_primary_key_is_mlbam_id():
    players = Base.metadata.tables["warehouse.players"]
    assert [c.name for c in players.primary_key.columns] == ["player_id"]
