"""Alembic environment.

The target metadata is ``mlbsim_core.Base.metadata``; importing
``mlbsim_data.models`` registers every ORM table on it. The database URL comes
from ``mlbsim_core`` settings unless one was already set on the config (e.g. by
``mlbsim db upgrade`` or an explicit ``-x url=``).
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

import mlbsim_data.models  # noqa: F401  -- registers tables on Base.metadata
from mlbsim_core.config import get_settings
from mlbsim_core.db import SCHEMAS, Base

config = context.config

if not config.get_main_option("sqlalchemy.url"):
    # Escape '%' for configparser interpolation (percent-encoded passwords).
    config.set_main_option("sqlalchemy.url", str(get_settings().database_url).replace("%", "%%"))

target_metadata = Base.metadata


def _include_name(name: str | None, type_: str, _parent: dict) -> bool:
    # Only manage our own schemas; ignore anything else in the database.
    if type_ == "schema":
        return name in SCHEMAS
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        include_schemas=True,
        include_name=_include_name,
        compare_type=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            include_name=_include_name,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
