"""Generate and store game predictions for a date from the trained direct models."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import delete, insert, select

from mlbsim_core import get_logger, get_settings, session_scope
from mlbsim_data.models import Game, GameFeature, GamePrediction
from mlbsim_features import feature_vector
from mlbsim_features.pipeline import build_game_features
from mlbsim_models.direct import RunsPoisson, WinLogit, derive_game_probs, score_grid
from mlbsim_models.explain import linear_shap, top_factors
from mlbsim_models.pipelines.train import MODEL_ID

_log = get_logger(__name__)


@dataclass(slots=True)
class PredictSummary:
    date: str
    model_id: str
    games: int
    written: int


def _load_models() -> tuple[WinLogit, RunsPoisson, RunsPoisson]:
    adir = get_settings().data_dir / "artifacts"
    return (
        WinLogit.load(adir / f"{MODEL_ID}.win_logit.joblib"),
        RunsPoisson.load(adir / f"{MODEL_ID}.runs_home.joblib"),
        RunsPoisson.load(adir / f"{MODEL_ID}.runs_away.joblib"),
    )


def _diff_features_for_date(date: str, feature_set_id: str) -> dict[int, dict[str, float]]:
    stmt = (
        select(GameFeature.game_pk, GameFeature.features)
        .join(Game, Game.game_pk == GameFeature.game_pk)
        .where(
            GameFeature.feature_set_id == feature_set_id,
            GameFeature.side == "diff",
            Game.game_date == dt.date.fromisoformat(date),
        )
    )
    with session_scope() as s:
        return {r.game_pk: r.features for r in s.execute(stmt)}


def predict_date(
    date: str,
    *,
    feature_set_id: str = "fs_v1",
    ensure_features: bool = True,
    rebuild: bool = True,
) -> PredictSummary:
    """Write one ``game_predictions`` row per game on ``date`` from ``direct_v1``."""
    with session_scope() as s:
        game_pks = list(
            s.scalars(
                select(Game.game_pk).where(
                    Game.game_date == dt.date.fromisoformat(date), Game.game_type == "R"
                )
            )
        )
    if not game_pks:
        return PredictSummary(date=date, model_id=MODEL_ID, games=0, written=0)

    if ensure_features:
        build_game_features(game_pks=game_pks, feature_set_id=feature_set_id)

    feats = _diff_features_for_date(date, feature_set_id)
    win, runs_home, runs_away = _load_models()
    now = dt.datetime.now(dt.UTC)

    rows: list[dict[str, Any]] = []
    for game_pk in game_pks:
        diff = feats.get(game_pk)
        if diff is None:
            continue
        xv = feature_vector(diff)
        x = [xv]
        p_home = float(win.predict_proba(x)[0])
        mu_h = float(runs_home.predict_mu(x)[0])
        mu_a = float(runs_away.predict_mu(x)[0])
        d = derive_game_probs(score_grid(mu_h, mu_a))
        contribs, base_logit = linear_shap(win, xv)
        factors = top_factors(contribs, base_logit, k=4)
        rows.append(
            {
                "game_pk": game_pk,
                "model_id": MODEL_ID,
                "created_at": now,
                "is_live": False,
                "home_win_prob": round(p_home, 5),
                "away_win_prob": round(1.0 - p_home, 5),
                "exp_home_runs": d["exp_home_runs"],
                "exp_away_runs": d["exp_away_runs"],
                "home_score_dist": d["home_score_dist"],
                "away_score_dist": d["away_score_dist"],
                "total_runs_dist": d["total_runs_dist"],
                "run_diff_dist": None,
                "p_extra_innings": d["p_extra_innings"],
                "p_shutout_home": d["p_shutout_home"],
                "p_shutout_away": d["p_shutout_away"],
                "p_one_run_game": d["p_one_run_game"],
                "most_likely_scores": d["most_likely_scores"],
                "factors": {
                    "win_source": "logistic",
                    "runs_source": "poisson",
                    "mu_home": round(mu_h, 3),
                    "mu_away": round(mu_a, 3),
                    "poisson_home_win_prob": d["home_win_prob"],
                    "top_factors": factors,
                },
            }
        )

    with session_scope() as s:
        if rebuild:
            s.execute(
                delete(GamePrediction).where(
                    GamePrediction.model_id == MODEL_ID,
                    GamePrediction.game_pk.in_(game_pks),
                )
            )
        if rows:
            s.execute(insert(GamePrediction), rows)

    summary = PredictSummary(date=date, model_id=MODEL_ID, games=len(game_pks), written=len(rows))
    _log.info("predict.date", **asdict(summary))
    return summary
