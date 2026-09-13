"""Reconcile player identities from the Chadwick register into ``warehouse.players``.

This is the M0 end-to-end slice: fetch ``people.csv`` -> parse -> idempotent
upsert. Roster-derived attributes (bats/throws/position) are filled in M1 from
the MLB Stats API and must not be clobbered here.
"""

from __future__ import annotations

from mlbsim_core import get_logger, session_scope
from mlbsim_data.loaders import upsert
from mlbsim_data.models import Player
from mlbsim_data.sources.chadwick import PlayerXRef, fetch_people

_log = get_logger(__name__)

# Only columns the register is authoritative for. Notably excludes
# bats/throws/primary_pos so M1 roster loads are not overwritten.
_UPDATE_COLUMNS = ("retro_id", "fg_id", "bbref_id", "full_name", "birth_date", "debut_date")


def _to_row(rec: PlayerXRef) -> dict[str, object]:
    return {
        "player_id": rec.player_id,
        "retro_id": rec.retro_id,
        "fg_id": rec.fg_id,
        "bbref_id": rec.bbref_id,
        "full_name": rec.full_name,
        "birth_date": rec.birth_date,
        "debut_date": None,  # exact debut date comes from Stats API in M1
    }


def load_crosswalk(*, use_cache: bool = True) -> int:
    """Fetch the register and upsert every MLB player. Returns rows written."""
    records = fetch_people(use_cache=use_cache)
    rows = [_to_row(r) for r in records]
    with session_scope() as session:
        written = upsert(
            session,
            Player,
            rows,
            index_elements=["player_id"],
            update_columns=_UPDATE_COLUMNS,
        )
    _log.info("crosswalk.loaded", rows=written)
    return written
