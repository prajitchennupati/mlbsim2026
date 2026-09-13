"""Step-runner plumbing shared by the flows."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import requests

from mlbsim_core import get_logger, get_settings

_log = get_logger(__name__)


@dataclass(slots=True)
class StepResult:
    name: str
    ok: bool
    detail: str = ""
    elapsed_ms: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "elapsed_ms": self.elapsed_ms,
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass(slots=True)
class FlowRun:
    """Accumulates :class:`StepResult` rows and renders the flow summary."""

    flow: str
    started_at: float = field(default_factory=time.time)
    steps: list[StepResult] = field(default_factory=list)

    def step(self, name: str, fn: Callable[[], Any], *, required: bool = False) -> Any:
        """Run ``fn``; record timing + a detail string; swallow and log exceptions."""
        t0 = time.perf_counter()
        try:
            out = fn()
            res = StepResult(
                name=name,
                ok=True,
                detail=_detail(out),
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
            )
            _log.info("flow.step.ok", flow=self.flow, step=name, detail=res.detail)
        except Exception as exc:
            res = StepResult(
                name=name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
            )
            _log.error(
                "flow.step.fail",
                flow=self.flow,
                step=name,
                error=res.error,
                trace=traceback.format_exc(),
            )
            out = None
        self.steps.append(res)
        if required and not res.ok:
            raise FlowAbortedError(self.flow, name, res.error or "step failed")
        return out

    def summary(self) -> dict[str, Any]:
        ok = all(s.ok for s in self.steps)
        return {
            "flow": self.flow,
            "ok": ok,
            "elapsed_ms": int((time.time() - self.started_at) * 1000),
            "n_steps": len(self.steps),
            "n_failed": sum(1 for s in self.steps if not s.ok),
            "steps": [s.as_dict() for s in self.steps],
        }


class FlowAbortedError(RuntimeError):
    def __init__(self, flow: str, step: str, reason: str) -> None:
        super().__init__(f"{flow}: required step {step!r} failed: {reason}")
        self.flow, self.step, self.reason = flow, step, reason


def _detail(out: Any) -> str:
    if out is None:
        return "ok"
    if isinstance(out, str):
        return out[:200]
    if hasattr(out, "as_dict"):
        return _detail(out.as_dict())
    if hasattr(out, "__dict__") and out.__dict__:
        return ", ".join(f"{k}={v}" for k, v in out.__dict__.items())[:200]
    if isinstance(out, dict):
        flat = {k: v for k, v in out.items() if isinstance(v, (int, float, str, bool))}
        return ", ".join(f"{k}={v}" for k, v in flat.items())[:200] or "ok"
    return str(out)[:200]


def ping_revalidate() -> str:
    """POST to the frontend on-demand revalidation webhook, if one is configured."""
    st = get_settings()
    if not st.web_revalidate_url:
        return "revalidate webhook not configured; skipped"
    resp = requests.post(
        st.web_revalidate_url,
        json={
            "secret": st.revalidate_secret,
            "paths": ["/", "/live", "/games", "/predictions", "/standings", "/playoffs", "/teams"],
        },
        timeout=st.http_timeout_seconds,
    )
    resp.raise_for_status()
    return f"revalidate ping -> {resp.status_code}"


def alert(summary: dict[str, Any]) -> None:
    """Best-effort failure notification to a Slack/Discord-style webhook."""
    st = get_settings()
    if not st.alert_webhook_url or summary.get("ok"):
        return
    failed = [s["name"] for s in summary.get("steps", []) if not s.get("ok")]
    text = f":rotating_light: `{summary['flow']}` failed steps: {', '.join(failed) or 'unknown'}"
    try:
        requests.post(
            st.alert_webhook_url,
            json={"text": text, "content": text},
            timeout=st.http_timeout_seconds,
        ).raise_for_status()
    except Exception as exc:
        _log.warning("flow.alert.fail", error=f"{type(exc).__name__}: {exc}")
