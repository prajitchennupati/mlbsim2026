from __future__ import annotations

from typer.testing import CliRunner

from mlbsim_pipeline.cli import app
from mlbsim_pipeline.paths import alembic_ini, repo_root

runner = CliRunner()


def test_version_lists_every_component():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    for pkg in (
        "mlbsim_core",
        "mlbsim_data",
        "mlbsim_features",
        "mlbsim_models",
        "mlbsim_engine",
        "mlbsim_pipeline",
    ):
        assert pkg in result.stdout


def test_paths_resolve_to_repo():
    assert (repo_root() / "pyproject.toml").is_file()
    assert alembic_ini().name == "alembic.ini"


def test_help_shows_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "db" in result.stdout
    assert "crosswalk" in result.stdout
