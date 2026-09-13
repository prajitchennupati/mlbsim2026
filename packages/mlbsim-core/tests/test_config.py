from __future__ import annotations

import importlib

import mlbsim_core.config as config_mod
from mlbsim_core import get_settings


def _fresh_settings(monkeypatch, **env):
    monkeypatch.delenv("MLBSIM_DATABASE_URL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    importlib.reload(config_mod)
    return config_mod.Settings()


def test_defaults_are_sane(monkeypatch):
    monkeypatch.chdir("/")  # ensure no stray .env is picked up from cwd
    settings = _fresh_settings(monkeypatch)
    assert settings.environment == "local"
    assert settings.database_url.scheme.startswith("postgresql")
    assert settings.landing_dir == settings.data_dir / "landing"
    assert settings.is_deployed is False


def test_env_prefix_override(monkeypatch):
    settings = _fresh_settings(
        monkeypatch,
        MLBSIM_ENVIRONMENT="production",
        MLBSIM_LOG_LEVEL="DEBUG",
    )
    assert settings.environment == "production"
    assert settings.log_level == "DEBUG"
    assert settings.is_deployed is True


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
