"""The ``mlbsim`` command-line interface.

M0 surface:
    mlbsim version              show component versions and resolved config
    mlbsim db upgrade [rev]     apply Alembic migrations
    mlbsim db current           show the current migration revision
    mlbsim crosswalk            fetch the Chadwick register into warehouse.players
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from mlbsim_core import configure_logging, get_logger, get_settings

if TYPE_CHECKING:
    from alembic.config import Config

app = typer.Typer(add_completion=False, help="mlbplayoffs2026 control CLI")
db_app = typer.Typer(help="Database migrations")
ingest_app = typer.Typer(help="Ingest MLB source data into the warehouse")
transform_app = typer.Typer(help="Rebuild derived aggregates from warehouse facts")
features_app = typer.Typer(help="Point-in-time feature store")
elo_app = typer.Typer(help="Elo team ratings baseline")
ensemble_app = typer.Typer(help="Stacked + calibrated ensemble")
flow_app = typer.Typer(help="Orchestration flows (daily / live / status)")
query_app = typer.Typer(help="Inspect ingested data")
app.add_typer(db_app, name="db")
app.add_typer(ingest_app, name="ingest")
app.add_typer(transform_app, name="transform")
app.add_typer(features_app, name="features")
app.add_typer(elo_app, name="elo")
app.add_typer(ensemble_app, name="ensemble")
app.add_typer(flow_app, name="flow")
app.add_typer(query_app, name="query")

_log = get_logger(__name__)


@app.callback()
def _main() -> None:
    configure_logging()


@app.command()
def version() -> None:
    """Print component versions and the resolved environment."""
    import mlbsim_core
    import mlbsim_data
    import mlbsim_engine
    import mlbsim_features
    import mlbsim_models
    import mlbsim_pipeline

    settings = get_settings()
    typer.echo(f"mlbplayoffs2026  (environment: {settings.environment})")
    for mod in (
        mlbsim_core,
        mlbsim_data,
        mlbsim_features,
        mlbsim_models,
        mlbsim_engine,
        mlbsim_pipeline,
    ):
        typer.echo(f"  {mod.__name__:<18} {mod.__version__}")


def _alembic_config() -> Config:
    from alembic.config import Config

    from mlbsim_pipeline.paths import alembic_ini

    cfg = Config(str(alembic_ini()))
    # configparser treats '%' as interpolation; escape it so URLs with a
    # percent-encoded password (e.g. '%3F') survive set_main_option.
    cfg.set_main_option("sqlalchemy.url", str(get_settings().database_url).replace("%", "%%"))
    return cfg


@db_app.command("upgrade")
def db_upgrade(revision: str = typer.Argument("head")) -> None:
    """Apply migrations up to ``revision`` (default: head)."""
    from alembic import command

    command.upgrade(_alembic_config(), revision)
    _log.info("db.upgrade.done", revision=revision)


@db_app.command("downgrade")
def db_downgrade(revision: str = typer.Argument(..., help="Target revision, or 'base'.")) -> None:
    """Revert migrations down to ``revision`` (uses the configured MLBSIM_DATABASE_URL)."""
    from alembic import command

    command.downgrade(_alembic_config(), revision)
    _log.info("db.downgrade.done", revision=revision)


@db_app.command("current")
def db_current() -> None:
    """Show the current migration revision."""
    from alembic import command

    command.current(_alembic_config(), verbose=True)


@app.command("crosswalk")
def crosswalk_load(
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore today's landing-zone copy and re-download."
    ),
) -> None:
    """Load the Chadwick Bureau player-ID crosswalk into warehouse.players."""
    from mlbsim_data.crosswalk import load_crosswalk

    written = load_crosswalk(use_cache=not refresh)
    typer.echo(f"upserted {written} players")


@ingest_app.command("schedule")
def ingest_schedule(
    start: str = typer.Argument(..., help="Start date YYYY-MM-DD"),
    end: str | None = typer.Argument(None, help="End date YYYY-MM-DD (default: start)"),
) -> None:
    """Ingest games / teams / weather / probables for a date range."""
    from mlbsim_data.ingest import ingest_schedule_range

    s = ingest_schedule_range(start, end)
    typer.echo(f"{s.games_loaded}/{s.games_seen} games loaded for {s.start}..{s.end}")


@ingest_app.command("game")
def ingest_one_game(
    game_pk: int = typer.Argument(..., help="MLB gamePk"),
    final: bool = typer.Option(
        False, "--final", help="Treat the game as completed (cache its feed permanently)."
    ),
) -> None:
    """Ingest one game's full detail (PAs, pitches, lineups, game logs)."""
    from mlbsim_data.ingest import ingest_game

    s = ingest_game(game_pk, assume_final=final)
    typer.echo(
        f"game {s.game_pk} [{s.status}] pa={s.plate_appearances} pitches={s.pitches} "
        f"batting_logs={s.batting_logs} pitching_logs={s.pitching_logs}"
    )


@ingest_app.command("season")
def ingest_full_season(
    year: int = typer.Argument(..., help="Season year, e.g. 2024"),
    include_incomplete: bool = typer.Option(
        False, "--include-incomplete", help="Also ingest games that are not yet Final."
    ),
    skip_statcast: bool = typer.Option(False, "--skip-statcast"),
) -> None:
    """Ingest a whole season: schedule, then every game's detail, then Statcast."""
    from mlbsim_data.ingest import ingest_season

    s = ingest_season(year, only_final=not include_incomplete, with_statcast=not skip_statcast)
    typer.echo(f"season {year}: {len(s.game_pks)} games scheduled, detail ingest complete")


@ingest_app.command("statcast")
def ingest_statcast(
    start: str = typer.Argument(..., help="Start date YYYY-MM-DD"),
    end: str | None = typer.Argument(None, help="End date YYYY-MM-DD (default: start)"),
) -> None:
    """Enrich batted_balls with Baseball Savant expected stats over a date range."""
    from mlbsim_data.ingest import ingest_statcast_range

    n = ingest_statcast_range(start, end or start)
    typer.echo(f"enriched {n} batted balls for {start}..{end or start}")


@transform_app.command("season-stats")
def transform_season_stats(
    season: int = typer.Argument(..., help="Season year"),
    through: str | None = typer.Option(
        None, "--through", help="Point-in-time cutoff YYYY-MM-DD (default: full season)."
    ),
) -> None:
    """Rebuild player_season_stats and team_season_stats for a season."""
    from mlbsim_data.transform import rebuild_season_stats, rebuild_team_season_stats

    players = rebuild_season_stats(season, through_date=through)
    teams = rebuild_team_season_stats(season, through_date=through)
    typer.echo(f"season {season}: {players} player rows, {teams} team rows")


@elo_app.command("build")
def elo_build(
    seasons: str | None = typer.Option(
        None, "--seasons", help="Comma-separated seasons, e.g. 2021,2022,2023 (default: all)."
    ),
) -> None:
    """Run Elo over finished games; persist elo_ratings + elo_v1 game predictions."""
    from mlbsim_models.pipelines import build_elo

    yrs = [int(x) for x in seasons.split(",")] if seasons else None
    s = build_elo(seasons=yrs)
    typer.echo(
        f"{s.model_id}: {s.games_processed} games, {s.rating_rows} rating rows, "
        f"{s.prediction_rows} predictions"
    )


@features_app.command("build")
def features_build(
    seasons: str | None = typer.Option(None, "--seasons", help="Comma-separated, e.g. 2022,2023."),
) -> None:
    """Compute home/away/diff game feature rows for the given seasons (default: all)."""
    from mlbsim_features.pipeline import build_game_features

    yrs = [int(x) for x in seasons.split(",")] if seasons else None
    n = build_game_features(seasons=yrs)
    typer.echo(f"wrote {n} game_features rows")


@app.command("train")
def train(
    through: str = typer.Option(
        ..., "--through", help="Train on games strictly before YYYY-MM-DD."
    ),
    start: str | None = typer.Option(None, "--start", help="Earliest training date YYYY-MM-DD."),
) -> None:
    """Train the direct models (logistic win prob + Poisson runs) and register direct_v1."""
    from mlbsim_models.pipelines import train_direct_models

    try:
        s = train_direct_models(train_end=through, train_start=start)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{s.model_id}: trained on {s.n_train} games (< {s.train_end})")
    for k, v in s.in_sample.items():
        typer.echo(f"  {k}={v}")


@app.command("predict")
def predict(
    date: str = typer.Argument(..., help="Game date YYYY-MM-DD"),
) -> None:
    """Build features + write direct_v1 predictions for every game on a date."""
    from mlbsim_models.pipelines import predict_date

    s = predict_date(date)
    typer.echo(f"{s.date}: {s.written}/{s.games} games predicted ({s.model_id})")


@app.command("train-gbm")
def train_gbm_cmd(
    through: str = typer.Option(..., "--through", help="Train on games before YYYY-MM-DD."),
    start: str | None = typer.Option(None, "--start", help="Earliest training date YYYY-MM-DD."),
) -> None:
    """Train the gradient-boosted win model and register gbm_v1 (ensemble base model)."""
    from mlbsim_models.pipelines import train_gbm

    try:
        s = train_gbm(train_end=through, train_start=start)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{s.model_id}: trained on {s.n_train} games (< {s.train_end})")
    for k, v in s.in_sample.items():
        typer.echo(f"  {k}={v}")


@app.command("predict-gbm")
def predict_gbm_cmd(date: str = typer.Argument(..., help="Game date YYYY-MM-DD")) -> None:
    """Write gbm_v1 predictions for every game on a date."""
    from mlbsim_models.pipelines import predict_gbm_date

    s = predict_gbm_date(date)
    typer.echo(f"{s['date']}: {s['written']}/{s['games']} games predicted ({s['model_id']})")


@app.command("project")
def project_cmd(
    player_id: int = typer.Argument(..., help="MLBAM player id"),
    season: int = typer.Option(..., "--season", help="Target season, e.g. 2026"),
    pitcher: bool = typer.Option(False, "--pitcher"),
) -> None:
    """Print a Marcel projection for one player from their prior seasons."""
    from mlbsim_models.pipelines import marcel_for_player

    p = marcel_for_player(player_id, season, is_pitcher=pitcher)
    typer.echo(f"player {player_id} -> {season}: proj_pa={p.proj_pa} age_factor={p.age_factor}")
    for name, v in p.components.items():
        typer.echo(f"  {name:<10} {v:.4f}")


@app.command("projections-build")
def projections_build(
    season: int = typer.Argument(..., help="Target season"),
) -> None:
    """Build + store Marcel projections (player_season_stats split='marcel')."""
    from mlbsim_models.pipelines import store_marcel_projections

    n = store_marcel_projections(season)
    typer.echo(f"stored {n} Marcel projections for {season}")


@app.command("predict-players")
def predict_players(
    game_pk: int = typer.Argument(..., help="MLB gamePk"),
    n: int = typer.Option(10_000, "--n"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Simulate a game and write per-batter / per-starter prediction rows (sim_v1)."""
    from mlbsim_models.pipelines import predict_players_for_game

    try:
        s = predict_players_for_game(game_pk, n_sims=n, seed=seed)
    except LookupError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"game {game_pk}: {s.batters} batter + {s.pitchers} pitcher predictions "
        f"(sim_run {s.sim_run_id})"
    )


@ensemble_app.command("train")
def ensemble_train(
    through: str = typer.Option(
        ..., "--through", help="Train on games strictly before YYYY-MM-DD."
    ),
) -> None:
    """Fit the stacked meta-learner + isotonic calibrator over the base win models."""
    from mlbsim_models.pipelines import train_ensemble

    try:
        s = train_ensemble(train_end=through)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{s.model_id}: fit on {s.n_train}, calibrated on {s.n_calibration}")
    typer.echo(f"  weights: {s.weights}")
    for k, v in s.in_sample.items():
        typer.echo(f"  {k}={v}")


@ensemble_app.command("predict")
def ensemble_predict(date: str = typer.Argument(..., help="Game date YYYY-MM-DD")) -> None:
    """Blend + calibrate the base predictions for a date's games (ensemble_v1)."""
    from mlbsim_models.pipelines import predict_ensemble_date

    s = predict_ensemble_date(date)
    typer.echo(f"{s['date']}: {s['written']}/{s['games']} games ({s['model_id']})")


@app.command("explain")
def explain_cmd(
    game_pk: int = typer.Argument(..., help="MLB gamePk"),
    model: str = typer.Option("direct_v1", "--model"),
    backend: str = typer.Option("template", "--backend", help="template | anthropic"),
) -> None:
    """Show the top factors + a natural-language explanation for a stored prediction."""
    from mlbsim_models.pipelines import explain_game

    try:
        out = explain_game(game_pk, model_id=model, backend=backend)
    except LookupError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    c = out["context"]
    typer.echo(f"{c['away']} @ {c['home']}  —  home win {c['home_win_prob']:.1%}")
    for f in out["factors"]:
        typer.echo(f"  {f['label']:<40} favours {f['favours']:<5} (Δp {f['prob_shift']:+.3f})")
    typer.echo(f"\n{out['explanation']}")


@app.command("simulate-season")
def simulate_season_cmd(  # noqa: PLR0917
    season: int = typer.Argument(..., help="Season year, e.g. 2026"),
    n: int = typer.Option(100_000, "--n", help="Number of simulated seasons."),
    seed: int = typer.Option(0, "--seed"),
    model: str = typer.Option("direct_v1", "--model", help="Per-game win-prob model."),
    as_of: str | None = typer.Option(None, "--as-of", help="Standings cutoff YYYY-MM-DD."),
    no_persist: bool = typer.Option(False, "--no-persist"),
) -> None:
    """Monte-Carlo the rest of a season: playoff / pennant / World Series odds."""
    from mlbsim_models.pipelines import simulate_season_from_db

    try:
        out = simulate_season_from_db(
            season, n_sims=n, seed=seed, model_id=model, as_of=as_of, persist=not no_persist
        )
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    m = out["meta"]
    typer.echo(
        f"season {season}: {n} sims in {m['runtime_ms']} ms  "
        f"({m['remaining_games']} remaining, {m['games_with_prediction']} with {model})"
    )
    for r in out["top_world_series"]:
        typer.echo(
            f"  team {r['team_id']}: playoffs={r['p_playoffs']:.3f} div={r['p_division']:.3f} "
            f"pennant={r['p_pennant']:.3f} WS={r['p_world_series']:.3f} expW={r['exp_wins']:.1f}"
        )
    for rnd, d in out["series"].items():
        typer.echo(f"  {rnd}: exp_games={d['exp_games']} p_sweep={d['p_sweep']:.3f}")


@app.command("simulate-game")
def simulate_game_cmd(
    game_pk: int = typer.Argument(..., help="MLB gamePk"),
    n: int = typer.Option(10_000, "--n", help="Number of simulated games."),
    seed: int = typer.Option(0, "--seed"),
    no_persist: bool = typer.Option(False, "--no-persist"),
) -> None:
    """Simulate one game at the plate-appearance level and print the summary."""
    from mlbsim_engine.pipeline import simulate_game_from_db

    try:
        s = simulate_game_from_db(game_pk, n_sims=n, seed=seed, persist=not no_persist)
    except LookupError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        f"game {game_pk}: {n} sims in {s['runtime_ms']} ms  home_win={s['home_win_prob']:.3f}"
    )
    typer.echo(
        f"  exp score {s['exp_home_runs']:.2f} - {s['exp_away_runs']:.2f}  "
        f"P(extra)={s['p_extra_innings']:.3f}  P(1-run)={s['p_one_run_game']:.3f}  "
        f"P(SO h/a)={s['p_shutout_home']:.3f}/{s['p_shutout_away']:.3f}"
    )
    top = s["most_likely_scores"][0]
    typer.echo(f"  most likely: home {top['home']}, away {top['away']}  (p={top['p']:.3f})")


@app.command("evaluate")
def evaluate(
    model_id: str = typer.Argument("elo_v1", help="Model id to score."),
    season: int | None = typer.Option(None, "--season"),
    start: str | None = typer.Option(None, "--start", help="YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="YYYY-MM-DD"),
    no_persist: bool = typer.Option(False, "--no-persist"),
) -> None:
    """Score stored predictions vs coin-flip / home-team baselines."""
    from mlbsim_models.pipelines import evaluate_model

    try:
        m = evaluate_model(model_id, season=season, start=start, end=end, persist=not no_persist)
    except LookupError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    r = m["model"]
    typer.echo(f"{model_id}  [{m['split_name']}]  n={r['n']}")
    typer.echo(
        f"  acc={r['accuracy']:.4f}  log_loss={r['log_loss']:.4f}  "
        f"brier={r['brier']:.4f}  auc={r['roc_auc']:.4f}  ece={r['ece']:.4f}"
    )
    for name, b in m["baselines"].items():
        typer.echo(f"  vs {name:<10} log_loss={b['log_loss']:.4f}  brier={b['brier']:.4f}")
    typer.echo(
        f"  beats coin-flip: {m['beats_coin_flip_logloss']}  "
        f"beats home-team: {m['beats_home_team_logloss']}"
    )


@flow_app.command("daily")
def flow_daily(
    date: str | None = typer.Option(None, "--date", help="Results day (default: yesterday)."),
    slate: str | None = typer.Option(None, "--slate", help="Predict day (default: today)."),
    train: bool = typer.Option(False, "--train", help="Also retrain direct + ensemble models."),
    season_sims: int = typer.Option(50_000, "--season-sims"),
    no_publish: bool = typer.Option(False, "--no-publish", help="Skip the revalidation webhook."),
) -> None:
    """Run the full daily flow: ingest -> features -> predict -> simulate -> publish."""
    import json

    from mlbsim_pipeline.flows import run_daily

    out = run_daily(
        date, slate_date=slate, do_train=train, season_sims=season_sims, publish=not no_publish
    )
    typer.echo(json.dumps(out, indent=2))
    raise typer.Exit(0 if out["ok"] else 1)


@flow_app.command("live")
def flow_live(
    date: str | None = typer.Option(None, "--date", help="Slate day YYYY-MM-DD (default: today)."),
    n_sims: int = typer.Option(6_000, "--n"),
    no_publish: bool = typer.Option(False, "--no-publish"),
) -> None:
    """Refresh live in-game win probabilities for every in-progress game."""
    import json

    from mlbsim_pipeline.flows import run_live

    out = run_live(date, n_sims=n_sims, publish=not no_publish)
    typer.echo(json.dumps(out, indent=2))
    raise typer.Exit(0 if out["ok"] else 1)


@flow_app.command("status")
def flow_status() -> None:
    """Print a freshness / backlog snapshot of the warehouse + serving schema."""
    import json

    from mlbsim_pipeline.flows import pipeline_status

    typer.echo(json.dumps(pipeline_status(), indent=2))


@query_app.command("game")
def query_game(game_pk: int = typer.Argument(..., help="MLB gamePk")) -> None:
    """Print a summary of one ingested game and its plate appearances."""
    from mlbsim_data.query import render_game

    try:
        typer.echo(render_game(game_pk))
    except LookupError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    app()
