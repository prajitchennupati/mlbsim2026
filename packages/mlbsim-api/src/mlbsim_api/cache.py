"""Response cache: Redis when reachable, an in-process LRU otherwise.

The API is read-only over precomputed tables, so a short TTL is safe and the
in-process fallback keeps local dev and CI working without Redis.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any

from mlbsim_core import get_logger, get_settings

_log = get_logger(__name__)
_LOCAL_MAX = 512


class Cache:
    def __init__(self) -> None:
        self._redis: Any = None
        self._local: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._connect()

    def _connect(self) -> None:
        try:
            import redis

            client = redis.Redis.from_url(str(get_settings().redis_url), socket_connect_timeout=0.5)
            client.ping()
            self._redis = client
            _log.info("cache.redis.connected")
        except Exception:
            self._redis = None
            _log.info("cache.redis.unavailable", fallback="in-process LRU")

    def get(self, key: str) -> Any | None:
        if self._redis is not None:
            raw = self._redis.get(key)
            return json.loads(raw) if raw is not None else None
        hit = self._local.get(key)
        if hit is None:
            return None
        expires_at, payload = hit
        if expires_at < time.time():
            self._local.pop(key, None)
            return None
        self._local.move_to_end(key)
        return json.loads(payload)

    def set(self, key: str, value: Any, *, ttl: int) -> None:
        payload = json.dumps(value, default=str)
        if self._redis is not None:
            self._redis.setex(key, ttl, payload)
            return
        self._local[key] = (time.time() + ttl, payload)
        self._local.move_to_end(key)
        while len(self._local) > _LOCAL_MAX:
            self._local.popitem(last=False)
