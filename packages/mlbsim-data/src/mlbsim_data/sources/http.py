"""HTTP fetch helpers with retries and a raw landing zone.

Every response is written verbatim under ``settings.landing_dir/<namespace>/<key>``
before parsing, so any ingestion run is replayable and auditable offline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mlbsim_core import get_logger, get_settings

_log = get_logger(__name__)

RETRYABLE = (requests.ConnectionError, requests.Timeout, requests.HTTPError)


def landing_path(namespace: str, key: str, suffix: str) -> Path:
    """Return (and mkdir) the landing-zone path for ``namespace/key``."""
    safe_key = key.replace("/", "_").strip("_")
    path = get_settings().landing_dir / namespace / f"{safe_key}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _fresh(path: Path, max_age_seconds: float | None) -> bool:
    if not path.exists():
        return False
    if max_age_seconds is None:
        return True
    return (time.time() - path.stat().st_mtime) < max_age_seconds


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _request(url: str, params: dict[str, str] | None) -> requests.Response:
    settings = get_settings()
    resp = requests.get(
        url,
        params=params,
        timeout=settings.http_timeout_seconds,
        headers={"User-Agent": "mlbplayoffs2026/0.0 (research project)"},
    )
    resp.raise_for_status()
    return resp


def fetch_text(
    url: str,
    *,
    namespace: str,
    key: str,
    suffix: str = ".txt",
    params: dict[str, str] | None = None,
    max_age_seconds: float | None = None,
) -> str:
    """GET ``url`` as text, caching to the landing zone.

    ``max_age_seconds=None`` means an existing landing file never expires (use for
    immutable resources such as a completed game's feed).
    """
    path = landing_path(namespace, key, suffix)
    if _fresh(path, max_age_seconds):
        _log.debug("landing.hit", namespace=namespace, key=key)
        return path.read_text(encoding="utf-8")
    text = _request(url, params).text
    path.write_text(text, encoding="utf-8")
    _log.info("landing.write", namespace=namespace, key=key, bytes=len(text))
    return text


def fetch_json(
    url: str,
    *,
    namespace: str,
    key: str,
    params: dict[str, str] | None = None,
    max_age_seconds: float | None = None,
) -> Any:
    """GET ``url`` and parse JSON, caching the raw body to the landing zone."""
    return json.loads(
        fetch_text(
            url,
            namespace=namespace,
            key=key,
            suffix=".json",
            params=params,
            max_age_seconds=max_age_seconds,
        )
    )
