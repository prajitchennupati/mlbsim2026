"""Reusable column type annotations for ORM models."""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from sqlalchemy import DateTime, func
from sqlalchemy.orm import mapped_column

# All timestamps are stored timezone-aware (UTC), per docs/DATABASE.md.
ts = Annotated[dt.datetime, mapped_column(DateTime(timezone=True))]
ts_opt = Annotated[dt.datetime | None, mapped_column(DateTime(timezone=True))]

# Server-side "row written at" audit column.
ingested_ts = Annotated[
    dt.datetime,
    mapped_column(DateTime(timezone=True), server_default=func.now()),
]
