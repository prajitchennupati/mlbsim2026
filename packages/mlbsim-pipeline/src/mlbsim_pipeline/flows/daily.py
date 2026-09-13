"""The daily pipeline: ingest -> aggregate -> features -> predict -> simulate -> publish.

Designed to run once per morning against the previous day's completed games and
the current day's slate. Every step is individually guarded; the flow returns a
summary dict and (optionally) pings the frontend revalidation webhook.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from mlbsim_core import get_logger
from mlbsim_pipeline.flows._run import FlowRun, alert, ping_revalidate

_log = get_logger(__name__)


def _yesterday() -> str:
    return (dt.date.today() - dt.timedelta(days=1)).isoformat()


def run_daily(
    date: str | None = None,
    *,
    slate_date: str | None = None,
    do_train: bool = False,
    season_sims: int = 50_000,
    publish: bool = True,
) -> dict[str, Any]:
    """Run the full daily flow.

    Parameters
    ----------
    date:        the "results" day to ingest + resolve (default: yesterday).
    slate_date:  the day to predict (default: today).
    do_train:    also retrain the direct + ensemble models through ``slate_date``.
    season_sims: Monte-Carlo season count for the standings/WS refresh.
    publish:     ping the frontend revalidation webhook at the end.
    """
    results_day = date or _yesterday()
    slate = slate_date or dt.date.today().isoformat()
    season = int(slate[:4])
    run = FlowRun(flow="daily")
    _log.info("flow.daily.start", results_day=results_day, slate=slate, season=season)

    # 1. Ingest the results day (schedule + each finished game's detail) and the
    #    slate day (schedule only — those games have not been played yet).
    sched = run.step("ingest_schedule", lambda: _ingest_schedule(results_day))
    game_pks: list[int] = list(getattr(sched, "game_pks", []) or [])
    run.step("ingest_games", lambda: _ingest_games(game_pks))
    run.step("ingest_statcast", lambda: _ingest_statcast(results_day))
    if slate != results_day:
        run.step("ingest_slate_schedule", lambda: _ingest_schedule(slate))

    # 2. Rebuild point-in-time aggregates + features for the season.
    run.step("season_stats", lambda: _season_stats(season))
    run.step("features", lambda: _features(season))

    # 3. Ratings + (optional) model training. build_elo only scores games that
    #    already have a final score, so the upcoming slate needs its own pass —
    #    elo_v1 is otherwise silent on any game that hasn't been played yet.
    run.step("elo", lambda: _elo(season))
    run.step("elo_predict_slate", lambda: _elo_predict(slate))
    if do_train:
        run.step("train_direct", lambda: _train_direct(slate))
        run.step("train_gbm", lambda: _train_gbm(slate))
        run.step("train_ensemble", lambda: _train_ensemble(slate))

    # 4. Predict today's slate with each base model, then blend + calibrate.
    run.step("predict", lambda: _predict(slate))
    run.step("predict_gbm", lambda: _predict_gbm(slate))
    run.step("predict_ensemble", lambda: _predict_ensemble(slate))

    # 5. Score everything that finished; refresh the season Monte Carlo.
    run.step("resolve_outcomes", lambda: _resolve(season))
    run.step("season_sim", lambda: _season_sim(season, season_sims))

    # 6. Publish.
    if publish:
        run.step("revalidate", ping_revalidate)

    summary = run.summary()
    summary["results_day"], summary["slate"], summary["season"] = results_day, slate, season
    alert(summary)
    _log.info("flow.daily.done", **{k: summary[k] for k in ("ok", "n_failed", "elapsed_ms")})
    return summary


# --- individual steps (kept tiny; import lazily so a partial env still loads) ---


def _ingest_schedule(day: str) -> Any:
    from mlbsim_data.ingest import ingest_schedule_range

    return ingest_schedule_range(day, day)


def _ingest_games(game_pks: list[int]) -> str:
    from mlbsim_data.ingest import ingest_game

    loaded = skipped = failed = 0
    for pk in game_pks:
        try:
            s = ingest_game(pk, assume_final=True, require_final=True)
            if s.skipped:
                skipped += 1
            else:
                loaded += 1
        except Exception as exc:
            failed += 1
            _log.warning("flow.daily.game_fail", game_pk=pk, error=f"{type(exc).__name__}: {exc}")
    return f"{loaded} loaded, {skipped} not-final, {failed} errored (of {len(game_pks)})"


def _ingest_statcast(day: str) -> str:
    from mlbsim_data.ingest import ingest_statcast_range

    n = ingest_statcast_range(day, day)
    return f"{n} batted balls enriched"


def _season_stats(season: int) -> str:
    from mlbsim_data.transform import rebuild_season_stats, rebuild_team_season_stats

    players = rebuild_season_stats(season)
    teams = rebuild_team_season_stats(season)
    return f"{players} player rows, {teams} team rows"


def _features(season: int) -> str:
    from mlbsim_features.pipeline import build_game_features

    return f"{build_game_features(seasons=[season])} game_features rows"


def _elo(season: int) -> Any:
    from mlbsim_models.pipelines import build_elo

    return build_elo(seasons=[season])


def _elo_predict(slate: str) -> Any:
    from mlbsim_models.pipelines import predict_elo_date

    return predict_elo_date(slate)


def _train_direct(through: str) -> Any:
    from mlbsim_models.pipelines import train_direct_models

    return train_direct_models(train_end=through)


def _train_gbm(through: str) -> Any:
    from mlbsim_models.pipelines import train_gbm

    return train_gbm(train_end=through)


def _predict_gbm(slate: str) -> Any:
    from mlbsim_models.pipelines import predict_gbm_date

    return predict_gbm_date(slate)


def _train_ensemble(through: str) -> Any:
    from mlbsim_models.pipelines import train_ensemble

    return train_ensemble(train_end=through)


def _predict(slate: str) -> Any:
    from mlbsim_models.pipelines import predict_date

    return predict_date(slate)


def _predict_ensemble(slate: str) -> Any:
    from mlbsim_models.pipelines import predict_ensemble_date

    return predict_ensemble_date(slate)


def _resolve(season: int) -> Any:
    from mlbsim_models.pipelines import resolve_outcomes

    return resolve_outcomes(since=f"{season}-01-01")


def _season_sim(season: int, n_sims: int) -> Any:
    from mlbsim_models.pipelines import simulate_season_from_db

    out = simulate_season_from_db(season, n_sims=n_sims, model_id="ensemble_v1", persist=True)
    return out.get("meta", out)
