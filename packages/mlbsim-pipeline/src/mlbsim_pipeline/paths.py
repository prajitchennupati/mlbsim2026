"""Repo-relative path resolution for the CLI (alembic config, etc.)."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Walk up from this file until a directory containing ``db/alembic.ini`` is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "db" / "alembic.ini").is_file():
            return parent
    # Fallback: workspace layout is packages/<pkg>/src/<module>/paths.py
    return here.parents[4]


def alembic_ini() -> Path:
    return repo_root() / "db" / "alembic.ini"
