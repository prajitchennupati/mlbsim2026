"""Orchestration test for ingest_game with sources + loaders faked (no database)."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from mlbsim_data import ingest


@pytest.fixture
def fake_backend(monkeypatch, game_feed, game_boxscore):
    calls: list[tuple[str, int]] = []

    class FakeWH:
        def __getattr__(self, name):
            def _loader(_session, rows):
                calls.append((name, len(list(rows))))
                return len(list(rows))

            return _loader

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr(ingest, "wh", FakeWH())
    monkeypatch.setattr(ingest, "session_scope", fake_session_scope)
    monkeypatch.setattr(ingest.mlb_statsapi, "game_feed", lambda pk, **kw: game_feed)
    monkeypatch.setattr(ingest.mlb_statsapi, "boxscore", lambda pk, **kw: game_boxscore)
    return calls


def test_ingest_game_summary(fake_backend):
    summary = ingest.ingest_game(744914, assume_final=True)
    assert summary.game_pk == 744914
    assert summary.status == "Final"
    assert summary.plate_appearances == 33
    assert summary.pitches > 100
    assert summary.batting_logs == 19
    assert summary.pitching_logs == 7
    assert summary.skipped is False


def test_ingest_game_loads_fk_parents_first(fake_backend):
    ingest.ingest_game(744914, assume_final=True)
    order = [name for name, _ in fake_backend]
    for parent, child in [
        ("load_parks", "load_games"),
        ("load_teams", "load_games"),
        ("load_players", "load_games"),
        ("load_games", "load_plate_appearances"),
        ("load_plate_appearances", "load_pitches"),
    ]:
        assert order.index(parent) < order.index(child), f"{parent} must load before {child}"


def test_require_final_skips_live_game(monkeypatch, game_feed):
    live = {
        **game_feed,
        "gameData": {**game_feed["gameData"], "status": {"detailedState": "In Progress"}},
    }
    monkeypatch.setattr(ingest.mlb_statsapi, "game_feed", lambda pk, **kw: live)
    summary = ingest.ingest_game(744914, require_final=True)
    assert summary.skipped is True
    assert summary.plate_appearances == 0
