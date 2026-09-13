"""FastAPI dependencies: a request-scoped read-only DB session and the cache."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from mlbsim_api.cache import Cache
from mlbsim_core.db import get_engine

_cache = Cache()


def get_session() -> Iterator[Session]:
    """Yield a short-lived session; the API only ever reads."""
    with Session(get_engine(), expire_on_commit=False) as session:
        yield session


def get_cache() -> Cache:
    return _cache
