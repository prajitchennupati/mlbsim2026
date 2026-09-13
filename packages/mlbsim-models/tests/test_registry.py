from __future__ import annotations

from mlbsim_models import registry


class _Settings:
    def __init__(self, uri: str) -> None:
        self.mlflow_tracking_uri = uri


def test_log_model_run_is_a_noop_without_a_tracking_uri(monkeypatch):
    monkeypatch.setattr(registry, "get_settings", lambda: _Settings(""))
    assert registry.log_model_run("elo_v1", metrics={"log_loss": 0.66}) is None


def test_log_model_run_returns_none_when_mlflow_is_not_installed(monkeypatch):
    # mlflow is an optional extra and is not part of the test environment.
    monkeypatch.setattr(registry, "get_settings", lambda: _Settings("http://localhost:5000"))
    assert registry.log_model_run("elo_v1") is None
