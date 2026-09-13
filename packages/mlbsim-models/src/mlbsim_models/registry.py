"""Optional MLflow sink for model runs.

The authoritative record is ``serving.model_versions``; this mirrors it to MLflow
when ``MLBSIM_MLFLOW_TRACKING_URI`` is set, and is a no-op otherwise so nothing
in the pipeline depends on MLflow being available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlbsim_core import get_logger, get_settings

_log = get_logger(__name__)


def log_model_run(
    model_id: str,
    *,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    artifacts: list[str | Path] | None = None,
    tags: dict[str, str] | None = None,
) -> str | None:
    """Log one run to MLflow. Returns the run id, or ``None`` when MLflow is off."""
    uri = get_settings().mlflow_tracking_uri
    if not uri:
        _log.debug("registry.mlflow.disabled", model_id=model_id)
        return None
    try:
        import mlflow
    except ImportError:
        _log.warning("registry.mlflow.not_installed", hint="pip install mlflow")
        return None

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("mlbplayoffs2026")
    with mlflow.start_run(run_name=model_id) as run:
        for k, v in (params or {}).items():
            mlflow.log_param(k, v)
        for k, v in (metrics or {}).items():
            mlflow.log_metric(k, float(v))
        for path in artifacts or []:
            p = Path(path)
            if p.exists():
                mlflow.log_artifact(str(p))
        mlflow.set_tags({"model_id": model_id, **(tags or {})})
        _log.info("registry.mlflow.logged", model_id=model_id, run_id=run.info.run_id)
        return str(run.info.run_id)
