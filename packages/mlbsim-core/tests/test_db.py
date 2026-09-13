from __future__ import annotations

from mlbsim_core import Base
from mlbsim_core.db import SCHEMAS


def test_base_metadata_uses_naming_convention():
    assert Base.metadata.naming_convention["pk"] == "pk_%(table_name)s"


def test_logical_schemas_declared():
    assert SCHEMAS == ("warehouse", "serving")
