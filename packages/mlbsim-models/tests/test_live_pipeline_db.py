"""update_live_games against a live database, with the MLB Stats API stubbed out."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select, text

from mlbsim_core import session_scope
from mlbsim_data.models import Game, GamePrediction, ModelVersion
from mlbsim_models.pipelines import live as live_mod
from mlbsim_models.pipelines import update_live_games

pytestmark = pytest.mark.integration

_PK = 970001
_HOME, _AWAY = 301, 302


def _live_feed() -> dict:
    return {
        "gameData": {
            "status": {"abstractGameState": "Live"},
            "probablePitchers": {"home": {"id": 111}, "away": {"id": 222}},
        },
        "liveData": {
            "linescore": {
                "currentInning": 8,
                "isTopInning": True,
                "outs": 1,
                "teams": {"home": {"runs": 5}, "away": {"runs": 2}},
                "offense": {"first": {"id": 9}, "battingOrder": 3},
            },
            "plays": {"currentPlay": {"matchup": {"pitcher": {"id": 999}}, "count": {"outs": 1}}},
        },
    }


@pytest.fixture
def seeded(monkeypatch):
    with session_scope() as s:
        s.execute(text("TRUNCATE serving.model_versions CASCADE"))
        s.execute(text("TRUNCATE warehouse.teams, warehouse.games CASCADE"))
        s.execute(
            text(
                "INSERT INTO warehouse.teams (team_id, abbr, name) VALUES "
                f"({_HOME},'HOM','Home'),({_AWAY},'AWY','Away')"
            )
        )
        s.add(
            Game(
                game_pk=_PK,
                season=2026,
                game_date=dt.date(2026, 7, 4),
                game_type="R",
                status="In Progress",
                home_team_id=_HOME,
                away_team_id=_AWAY,
            )
        )

    monkeypatch.setattr(
        live_mod.mlb_statsapi,
        "schedule",
        lambda *a, **k: [{"gamePk": _PK, "status": {"abstractGameState": "Live"}}],
    )
    monkeypatch.setattr(live_mod.mlb_statsapi, "game_feed", lambda pk, **k: _live_feed())


def test_update_live_games_writes_an_is_live_prediction(seeded):
    summary = update_live_games("2026-07-04", n_sims=2000)
    assert summary.live_games == 1
    assert summary.written == 1

    with session_scope() as s:
        assert s.get(ModelVersion, "live_v1") is not None
        rows = list(
            s.execute(
                select(GamePrediction).where(
                    GamePrediction.game_pk == _PK, GamePrediction.is_live.is_(True)
                )
            ).scalars()
        )
        assert len(rows) == 1
        pred = rows[0]
        assert pred.model_id == "live_v1"
        # home leads 5-2 in the 8th -> heavy home favourite
        assert float(pred.home_win_prob) > 0.9
        assert pred.game_state_json["inning"] == 8
        assert pred.game_state_json["home_score"] == 5
        assert float(pred.home_win_prob) + float(pred.away_win_prob) == pytest.approx(1.0, abs=1e-4)


def test_no_live_games_writes_nothing(seeded, monkeypatch):
    monkeypatch.setattr(live_mod.mlb_statsapi, "schedule", lambda *a, **k: [])
    summary = update_live_games("2026-07-04")
    assert summary.live_games == 0
    assert summary.written == 0
    with session_scope() as s:
        n = s.execute(select(func.count()).select_from(GamePrediction)).scalar_one()
        assert n == 0
