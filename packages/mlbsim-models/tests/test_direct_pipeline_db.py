"""Feature build -> train -> predict -> evaluate against a live database."""

from __future__ import annotations

import datetime as dt
import random

import pytest
from sqlalchemy import func, select, text

from mlbsim_core import session_scope
from mlbsim_data.models import (
    Game,
    GameFeature,
    GamePrediction,
    ModelVersion,
    PitchingGameLog,
    TeamGameLog,
)
from mlbsim_features.pipeline import build_game_features
from mlbsim_models.pipelines import evaluate_model, predict_date, train_direct_models

pytestmark = pytest.mark.integration

_TEAMS = [(1, "AAA"), (2, "BBB"), (3, "CCC"), (4, "DDD")]
_STRENGTH = {1: 0.66, 2: 0.54, 3: 0.46, 4: 0.34}
_PITCHERS = {tid: 1000 + tid for tid in _STRENGTH}  # one nominal starter per team

_TABLES = [
    "game_features",
    "feature_sets",
    "game_predictions",
    "model_eval_runs",
    "calibration_bins",
    "model_versions",
    "pitching_game_logs",
    "team_game_logs",
    "games",
    "players",
    "teams",
]


@pytest.fixture
def synthetic_league():
    rng = random.Random(11)
    games, tgl, pgl = [], [], []
    pk = 800000
    for season in (2023, 2024):
        day = dt.date(season, 4, 1)
        for _ in range(140):
            h, a = rng.sample([1, 2, 3, 4], 2)
            ph = _STRENGTH[h] / (_STRENGTH[h] + _STRENGTH[a])
            home_won = rng.random() < ph
            hs, as_ = (
                (rng.randint(4, 8), rng.randint(1, 4))
                if home_won
                else (rng.randint(1, 4), rng.randint(4, 8))
            )
            games.append(
                {
                    "game_pk": pk,
                    "season": season,
                    "game_date": day,
                    "game_type": "R",
                    "status": "Final",
                    "home_team_id": h,
                    "away_team_id": a,
                    "home_score": hs,
                    "away_score": as_,
                    "home_sp_id": _PITCHERS[h],
                    "away_sp_id": _PITCHERS[a],
                    "dh_game_num": 1,
                    "is_doubleheader": False,
                    "scheduled_innings": 9,
                }
            )
            for tid, rf, ra, won, is_home in (
                (h, hs, as_, home_won, True),
                (a, as_, hs, not home_won, False),
            ):
                tgl.append(
                    {
                        "team_id": tid,
                        "game_pk": pk,
                        "season": season,
                        "opponent_id": a if tid == h else h,
                        "is_home": is_home,
                        "runs_for": rf,
                        "runs_against": ra,
                        "hits_for": 8,
                        "hits_against": 8,
                        "won": won,
                    }
                )
            for tid, er in ((h, as_), (a, hs)):
                pgl.append(
                    {
                        "player_id": _PITCHERS[tid],
                        "game_pk": pk,
                        "team_id": tid,
                        "season": season,
                        "is_start": True,
                        "bf": 24,
                        "outs": 18,
                        "h": 6,
                        "r": er,
                        "er": er,
                        "bb": 2,
                        "ibb": 0,
                        "hbp": 0,
                        "so": 6,
                        "hr": 1,
                        "pitches": 90,
                        "strikes": 60,
                        "got_win": False,
                        "got_loss": False,
                        "got_save": False,
                        "quality_start": er <= 3,
                    }
                )
            pk += 1
            day += dt.timedelta(days=1)

    with session_scope() as s:
        s.execute(
            text(
                "TRUNCATE "
                + ", ".join(
                    f"warehouse.{t}"
                    for t in _TABLES
                    if t
                    not in (
                        "model_eval_runs",
                        "calibration_bins",
                        "model_versions",
                        "game_predictions",
                    )
                )
                + " CASCADE"
            )
        )
        s.execute(text("TRUNCATE serving.model_versions CASCADE"))
        s.execute(
            text(
                "INSERT INTO warehouse.teams (team_id, abbr, name) VALUES "
                + ",".join(f"({t},'{a}','{a}')" for t, a in _TEAMS)
            )
        )
        s.execute(
            text(
                "INSERT INTO warehouse.players (player_id, full_name) VALUES "
                + ",".join(f"({p},'SP {p}')" for p in _PITCHERS.values())
            )
        )
        s.execute(Game.__table__.insert(), games)
        s.execute(TeamGameLog.__table__.insert(), tgl)
        s.execute(PitchingGameLog.__table__.insert(), pgl)
    yield


def test_end_to_end_direct_pipeline(synthetic_league):
    n_feat = build_game_features(seasons=[2023, 2024])
    assert n_feat == 280 * 3  # 3 sides per game

    with session_scope() as s:
        assert (
            s.scalar(
                select(func.count()).select_from(GameFeature).where(GameFeature.side == "diff")
            )
            == 280
        )

    summary = train_direct_models(train_end="2024-01-01")
    assert summary.n_train == 140  # 2023 only
    assert summary.in_sample["win_auc"] > 0.55

    with session_scope() as s:
        mv = s.get(ModelVersion, "direct_v1")
        assert mv is not None
        assert mv.train_end == dt.date(2024, 1, 1)
        assert mv.feature_set_id == "fs_v1"

    # predict a specific 2024 date
    with session_scope() as s:
        a_date = s.scalar(
            select(Game.game_date).where(Game.season == 2024).order_by(Game.game_date).limit(1)
        )
    ps = predict_date(a_date.isoformat())
    assert ps.written == ps.games > 0

    with session_scope() as s:
        pred = s.scalar(
            select(GamePrediction).where(GamePrediction.model_id == "direct_v1").limit(1)
        )
        assert 0.0 < float(pred.home_win_prob) < 1.0
        assert pred.exp_home_runs is not None
        assert pred.home_score_dist and pred.total_runs_dist
        assert pred.most_likely_scores


def test_direct_v1_evaluation_pipeline_produces_sane_metrics(synthetic_league):
    # This exercises train -> predict -> evaluate end to end and checks the
    # metrics are finite and in the coin-flip ballpark. It deliberately does NOT
    # assert the model beats coin flip: the "beats baseline" claim is only
    # meaningful on real data and is tracked in docs/EVALUATION.md.
    build_game_features(seasons=[2023, 2024])
    train_direct_models(train_end="2024-01-01")
    with session_scope() as s:
        dates = list(
            s.scalars(
                select(Game.game_date)
                .where(Game.season == 2024)
                .distinct()
                .order_by(Game.game_date)
            )
        )
    for d in dates[:80]:  # synthetic schedule is one game per day
        predict_date(d.isoformat())

    m = evaluate_model("direct_v1", season=2024)
    r = m["model"]
    assert r["n"] > 60
    assert 0.0 < r["log_loss"] < 1.0  # finite, not diverged
    assert 0.0 < r["brier"] < 0.5
    assert 0.3 <= r["roc_auc"] <= 0.7  # crude synthetic signal -> near chance out of sample
    assert set(m["baselines"]) == {"coin_flip", "home_team"}
