"""Typed application settings, loaded from environment / ``.env``.

Every process (pipeline, API, tests) reads configuration through :func:`get_settings`.
Nothing else in the codebase should call ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn, RedisDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Runtime configuration for the platform.

    Values are resolved (highest precedence first) from: real environment
    variables, then a ``.env`` file at the repo root, then the defaults below.
    """

    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="MLBSIM_",
        extra="ignore",
    )

    environment: str = Field(
        default="local",
        description="local | ci | staging | production",
    )
    log_level: str = Field(default="INFO")
    log_json: bool = Field(
        default=False,
        description="Emit structured JSON logs (true in deployed environments).",
    )

    # --- Storage -----------------------------------------------------------
    database_url: PostgresDsn = Field(
        default=PostgresDsn("postgresql+psycopg://mlbsim:mlbsim@localhost:5432/mlbsim"),
        description="SQLAlchemy URL for the warehouse + serving database.",
    )
    redis_url: RedisDsn = Field(
        default=RedisDsn("redis://localhost:6379/0"),
        description="Redis URL used by the API response cache.",
    )
    db_pool_size: int = Field(default=5, ge=1)
    db_echo: bool = Field(default=False)

    # --- Local paths -----------------------------------------------------------
    data_dir: Path = Field(
        default=_REPO_ROOT / "data",
        description="Root for the raw landing zone and intermediate Parquet.",
    )

    # --- External data sources ------------------------------------------------
    mlb_statsapi_base: str = Field(default="https://statsapi.mlb.com/api")
    chadwick_register_url: str = Field(
        default=(
            "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people.csv"
        ),
        description="Chadwick Bureau player-ID crosswalk (people.csv).",
    )
    http_timeout_seconds: float = Field(default=30.0, gt=0)
    http_max_retries: int = Field(default=4, ge=0)

    # --- Pipeline / MLOps ----------------------------------------------------
    web_revalidate_url: str = Field(
        default="",
        description="Frontend on-demand revalidation webhook; empty disables the ping.",
    )
    revalidate_secret: str = Field(default="")
    mlflow_tracking_uri: str = Field(
        default="",
        description="If set, model runs are also logged to MLflow.",
    )
    alert_webhook_url: str = Field(
        default="",
        description="Slack/Discord-style webhook for pipeline failure alerts; empty disables.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def landing_dir(self) -> Path:
        """Directory for verbatim external responses (the raw landing zone)."""
        return self.data_dir / "landing"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_deployed(self) -> bool:
        return self.environment in {"staging", "production"}


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    return Settings()
