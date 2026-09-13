"""gbm_v1 / mlp_v1 / ensemble_v1: a real gradient-boosted tree, a real small
neural net, and a stacked+calibrated ensemble over them plus elo_sp_v1 -- all
trained on Game- and Stats-API-derived features (team form + starting-pitcher
FIP), reusing elo_v1's and elo_sp_v1's already-stored pre-game ratings/
adjustments so this needs zero new network calls.

Chronological split, never random (project non-negotiable): base models train
on the earliest slice, the stacker fits on the next slice (never the base
models' own training rows), and everyone is scored honestly on a final held-
out slice nobody has touched.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, insert, select

from mlbsim_core import get_logger, get_settings, session_scope
from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import FINAL_STATUSES, Game, GamePrediction, ModelVersion
from mlbsim_models.direct import GBMWinModel, MLPWinModel
from mlbsim_models.ensemble import StackedWinModel, fit_calibrator, load_calibrator
from mlbsim_models.evaluate.metrics import accuracy, brier_score, log_loss
from mlbsim_models.ml.team_form import build_tracker, run_team_form
from mlbsim_models.pipelines.elo_sp import team_ratings_before

_log = get_logger(__name__)


def _artifact_dir() -> str:
    d = get_settings().data_dir / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _safe_shap(gbm: GBMWinModel, x: list[list[float]]) -> list[list[dict[str, Any]] | None]:
    """Real shap.TreeExplainer factors for gbm_v1, or None per row if the
    optional `shap` extra isn't installed (e.g. a deployed API environment
    that never needs it, since this only runs offline during training/predict)."""
    if not x:
        return []
    try:
        from mlbsim_models.explain.gbm_shap import gbm_shap_factors_batch

        result: list[list[dict[str, Any]] | None] = list(gbm_shap_factors_batch(gbm, x))
        return result
    except ImportError:
        _log.warning("ml_stack.shap_unavailable", n=len(x))
        return [None] * len(x)

GBM_ID = "gbm_v1"
MLP_ID = "mlp_v1"
ENSEMBLE_ID = "ensemble_v1"
BASE_IDS = (GBM_ID, MLP_ID, "elo_sp_v1")

FEATURE_NAMES = [
    "d_win_pct",
    "d_run_diff_pg",
    "d_win_pct_l10",
    "d_run_diff_pg_l10",
    "d_rest_days",
    "d_elo",
    "d_sp_adj",
]


def _feature_row(
    home_form: dict[str, float],
    away_form: dict[str, float],
    home_elo: float,
    away_elo: float,
    *,
    home_sp_adj: float,
    away_sp_adj: float,
) -> list[float]:
    return [
        home_form["win_pct"] - away_form["win_pct"],
        home_form["run_diff_pg"] - away_form["run_diff_pg"],
        home_form["win_pct_l10"] - away_form["win_pct_l10"],
        home_form["run_diff_pg_l10"] - away_form["run_diff_pg_l10"],
        home_form["rest_days"] - away_form["rest_days"],
        (home_elo - away_elo) / 100.0,
        home_sp_adj - away_sp_adj,
    ]


@dataclass(slots=True)
class _Row:
    game_pk: int
    game_date: dt.date
    x: list[float]
    y: int
    elo_sp_p: float


def _load_rows(season: int) -> list[_Row]:
    with session_scope() as s:
        games = list(
            s.execute(
                select(
                    Game.game_pk,
                    Game.game_date,
                    Game.home_team_id,
                    Game.away_team_id,
                    Game.home_score,
                    Game.away_score,
                )
                .where(
                    Game.season == season,
                    Game.game_type == "R",
                    Game.status.in_(FINAL_STATUSES),
                    Game.home_score.is_not(None),
                )
                .order_by(Game.game_date, Game.game_pk)
            )
        )
        elo_rows = {
            r.game_pk: r.factors
            for r in s.execute(
                select(GamePrediction.game_pk, GamePrediction.factors).where(
                    GamePrediction.model_id == "elo_v1", GamePrediction.is_live.is_(False)
                )
            )
        }
        sp_rows = {
            r.game_pk: r.factors
            for r in s.execute(
                select(GamePrediction.game_pk, GamePrediction.factors).where(
                    GamePrediction.model_id == "elo_sp_v1", GamePrediction.is_live.is_(False)
                )
            )
        }
        elo_sp_p = {
            r.game_pk: float(r.home_win_prob)
            for r in s.execute(
                select(GamePrediction.game_pk, GamePrediction.home_win_prob).where(
                    GamePrediction.model_id == "elo_sp_v1", GamePrediction.is_live.is_(False)
                )
            )
        }

    form_by_pk = {row.game_pk: row for row in run_team_form([dict(g._mapping) for g in games])}

    rows: list[_Row] = []
    for g in games:
        elo = elo_rows.get(g.game_pk)
        form = form_by_pk.get(g.game_pk)
        if elo is None or form is None or g.game_pk not in elo_sp_p:
            continue
        sp = sp_rows.get(g.game_pk) or {}
        x = _feature_row(
            form.home_form,
            form.away_form,
            float(elo["home_rating"]),
            float(elo["away_rating"]),
            home_sp_adj=float(sp.get("home_sp_adjustment", 0.0)),
            away_sp_adj=float(sp.get("away_sp_adjustment", 0.0)),
        )
        rows.append(
            _Row(
                game_pk=g.game_pk,
                game_date=g.game_date,
                x=x,
                y=1 if g.home_score > g.away_score else 0,
                elo_sp_p=elo_sp_p[g.game_pk],
            )
        )
    return rows


def _score(preds: list[float], ys: list[int], label: str) -> dict[str, float]:
    metrics = {
        "n": len(ys),
        "accuracy": accuracy(ys, preds),
        "brier": brier_score(ys, preds),
        "log_loss": log_loss(ys, preds),
    }
    _log.info("ml_stack.score", label=label, **metrics)
    return metrics


@dataclass(slots=True)
class MlStackSummary:
    n_train: int
    n_stack: int
    n_eval: int
    gbm: dict[str, float]
    mlp: dict[str, float]
    ensemble: dict[str, float]
    elo_sp_baseline: dict[str, float]


def train_ml_stack(
    season: int, *, train_frac: float = 0.55, stack_frac: float = 0.25
) -> MlStackSummary:
    """Chronological 55/25/20 split: train base models / fit the stacker+
    calibrator / final honest eval. Registers gbm_v1, mlp_v1, ensemble_v1."""
    rows = _load_rows(season)
    if len(rows) < 300:
        raise ValueError(f"only {len(rows)} usable games -- need elo_v1/elo_sp_v1 run first")

    n = len(rows)
    n_train = int(n * train_frac)
    n_stack = int(n * (train_frac + stack_frac)) - n_train
    train = rows[:n_train]
    stack = rows[n_train : n_train + n_stack]
    evalset = rows[n_train + n_stack :]

    x_train = [r.x for r in train]
    y_train = [r.y for r in train]
    gbm = GBMWinModel(FEATURE_NAMES).fit(x_train, y_train)
    mlp = MLPWinModel(FEATURE_NAMES).fit(x_train, y_train)

    def base_probs(rs: list[_Row]) -> list[list[float]]:
        xs = [r.x for r in rs]
        g = gbm.predict_proba(xs)
        m = mlp.predict_proba(xs)
        e = [r.elo_sp_p for r in rs]
        return [list(t) for t in zip(g, m, e, strict=True)]

    stacker = StackedWinModel(list(BASE_IDS)).fit(base_probs(stack), [r.y for r in stack])
    stack_raw = list(stacker.predict_proba(base_probs(stack)))
    calibrator = fit_calibrator(stack_raw, [r.y for r in stack])

    eval_probs = base_probs(evalset)
    ys_eval = [r.y for r in evalset]
    gbm_eval = [p[0] for p in eval_probs]
    mlp_eval = [p[1] for p in eval_probs]
    elo_sp_eval = [p[2] for p in eval_probs]
    ensemble_eval = list(calibrator.transform(stacker.predict_proba(eval_probs)))

    now = dt.datetime.now(dt.UTC)
    with session_scope() as s:
        upsert(
            s,
            ModelVersion,
            [
                {
                    "model_id": GBM_ID,
                    "name": "Gradient-boosted trees (team form + SP FIP)",
                    "kind": "gbm",
                    "version": "1",
                    "trained_at": now,
                    "hyperparams_json": gbm.params,
                    "notes": "HistGradientBoostingClassifier on Game/Stats-API-derived features.",
                },
                {
                    "model_id": MLP_ID,
                    "name": "Neural net (MLP, team form + SP FIP)",
                    "kind": "mlp",
                    "version": "1",
                    "trained_at": now,
                    "hyperparams_json": {k: str(v) for k, v in mlp.params.items()},
                    "notes": "scikit-learn MLPClassifier on the same features as gbm_v1.",
                },
                {
                    "model_id": ENSEMBLE_ID,
                    "name": "Stacked ensemble (gbm_v1 + mlp_v1 + elo_sp_v1)",
                    "kind": "ensemble",
                    "version": "1",
                    "trained_at": now,
                    "hyperparams_json": {"base_models": list(BASE_IDS)},
                    "notes": "L2-logistic stack over base logits, isotonic/Platt calibrated.",
                },
            ],
            index_elements=["model_id"],
        )

    adir = _artifact_dir()
    gbm.save(f"{adir}/{GBM_ID}.joblib")
    mlp.save(f"{adir}/{MLP_ID}.joblib")
    stacker.save(f"{adir}/{ENSEMBLE_ID}.stack.joblib")
    calibrator.save(f"{adir}/{ENSEMBLE_ID}.calibrator.joblib")

    # Only ever persist/grade *out-of-sample* predictions -- gbm_v1/mlp_v1 never
    # saw `stack` or `evalset` during fit, and the stacker never saw `evalset`.
    stack_probs = base_probs(stack)
    stack_and_eval = stack + evalset
    gbm_all = [*[p[0] for p in stack_probs], *gbm_eval]
    mlp_all = [*[p[1] for p in stack_probs], *mlp_eval]
    gbm_shap = _safe_shap(gbm, [r.x for r in stack_and_eval])
    _write_predictions(
        [
            _prediction_row(r.game_pk, GBM_ID, now, p, factors)
            for r, p, factors in zip(stack_and_eval, gbm_all, gbm_shap, strict=True)
        ]
    )
    _write_predictions(
        [
            _prediction_row(r.game_pk, MLP_ID, now, p)
            for r, p in zip(stack_and_eval, mlp_all, strict=True)
        ]
    )
    _write_predictions(
        [
            _prediction_row(r.game_pk, ENSEMBLE_ID, now, p)
            for r, p in zip(evalset, ensemble_eval, strict=True)
        ]
    )

    summary = MlStackSummary(
        n_train=len(train),
        n_stack=len(stack),
        n_eval=len(evalset),
        gbm=_score(gbm_eval, ys_eval, "gbm_v1 (eval)"),
        mlp=_score(mlp_eval, ys_eval, "mlp_v1 (eval)"),
        ensemble=_score(ensemble_eval, ys_eval, "ensemble_v1 (eval)"),
        elo_sp_baseline=_score(elo_sp_eval, ys_eval, "elo_sp_v1 (same eval games)"),
    )
    _log.info(
        "ml_stack.train.done",
        n_train=summary.n_train,
        n_stack=summary.n_stack,
        n_eval=summary.n_eval,
    )
    return summary


def _prediction_row(
    game_pk: int,
    model_id: str,
    created_at: dt.datetime,
    p: float,
    top_factors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return {
        "game_pk": game_pk,
        "model_id": model_id,
        "created_at": created_at,
        "is_live": False,
        "home_win_prob": round(p, 5),
        "away_win_prob": round(1.0 - p, 5),
        "factors": {"top_factors": top_factors} if top_factors else None,
    }


def _write_predictions(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    by_model: dict[str, list[int]] = {}
    for r in rows:
        by_model.setdefault(r["model_id"], []).append(r["game_pk"])
    with session_scope() as s:
        for model_id, pks in by_model.items():
            s.execute(
                delete(GamePrediction).where(
                    GamePrediction.model_id == model_id, GamePrediction.game_pk.in_(pks)
                )
            )
        s.execute(insert(GamePrediction), rows)


@dataclass(slots=True)
class MlStackPredictSummary:
    gbm_rows: int
    mlp_rows: int
    ensemble_rows: int


def predict_ml_stack_date(season: int, start: str, end: str | None = None) -> MlStackPredictSummary:
    """Live/upcoming slate: load the saved gbm_v1/mlp_v1/ensemble_v1 and score
    [start, end] using each team's current form + elo_sp_v1's already-computed
    ratings/SP adjustments for those games (predict_elo_sp_date must run first)."""
    end = end or start
    start_d, end_d = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    adir = _artifact_dir()
    gbm = GBMWinModel.load(f"{adir}/{GBM_ID}.joblib")
    mlp = MLPWinModel.load(f"{adir}/{MLP_ID}.joblib")
    stacker = StackedWinModel.load(f"{adir}/{ENSEMBLE_ID}.stack.joblib")
    calibrator = load_calibrator(f"{adir}/{ENSEMBLE_ID}.calibrator.joblib")

    with session_scope() as s:
        all_games = list(
            s.execute(
                select(
                    Game.game_pk,
                    Game.game_date,
                    Game.home_team_id,
                    Game.away_team_id,
                    Game.home_score,
                    Game.away_score,
                ).where(Game.season == season, Game.game_type == "R")
            )
        )
        slate = [
            g
            for g in all_games
            if start_d <= g.game_date <= end_d and g.home_score is None and g.away_score is None
        ]
        sp_stmt = select(
            GamePrediction.game_pk, GamePrediction.factors, GamePrediction.home_win_prob
        ).where(
            GamePrediction.model_id == "elo_sp_v1",
            GamePrediction.game_pk.in_([g.game_pk for g in slate]),
        )
        sp_rows = {r.game_pk: (r.factors, float(r.home_win_prob)) for r in s.execute(sp_stmt)}

    if not slate:
        return MlStackPredictSummary(0, 0, 0)

    tracker = build_tracker([dict(g._mapping) for g in all_games])
    team_ids = {g.home_team_id for g in slate} | {g.away_team_id for g in slate}
    ratings = team_ratings_before(team_ids, start_d)

    now = dt.datetime.now(dt.UTC)
    xs, pks, elo_sp_ps = [], [], []
    for g in slate:
        sp = sp_rows.get(g.game_pk)
        if sp is None:
            continue  # elo_sp_v1 hasn't scored this game yet -- skip rather than guess
        factors, elo_sp_p = sp
        x = _feature_row(
            tracker.snapshot(g.home_team_id, g.game_date),
            tracker.snapshot(g.away_team_id, g.game_date),
            ratings.get(g.home_team_id, 1500.0),
            ratings.get(g.away_team_id, 1500.0),
            home_sp_adj=float((factors or {}).get("home_sp_adjustment", 0.0)),
            away_sp_adj=float((factors or {}).get("away_sp_adjustment", 0.0)),
        )
        xs.append(x)
        pks.append(g.game_pk)
        elo_sp_ps.append(elo_sp_p)

    if not xs:
        return MlStackPredictSummary(0, 0, 0)

    gbm_p = gbm.predict_proba(xs)
    mlp_p = mlp.predict_proba(xs)
    base = [[g, m, e] for g, m, e in zip(gbm_p, mlp_p, elo_sp_ps, strict=True)]
    ensemble_p = calibrator.transform(stacker.predict_proba(base))
    gbm_shap = _safe_shap(gbm, xs)

    _write_predictions(
        [
            _prediction_row(pk, GBM_ID, now, float(p), factors)
            for pk, p, factors in zip(pks, gbm_p, gbm_shap, strict=True)
        ]
    )
    _write_predictions(
        [_prediction_row(pk, MLP_ID, now, float(p)) for pk, p in zip(pks, mlp_p, strict=True)]
    )
    _write_predictions(
        [
            _prediction_row(pk, ENSEMBLE_ID, now, float(p))
            for pk, p in zip(pks, ensemble_p, strict=True)
        ]
    )
    return MlStackPredictSummary(gbm_rows=len(pks), mlp_rows=len(pks), ensemble_rows=len(pks))
