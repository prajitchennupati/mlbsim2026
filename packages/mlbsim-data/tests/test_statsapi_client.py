from __future__ import annotations

import mlbsim_data.sources.mlb_statsapi as api


def test_schedule_flattens_dates(monkeypatch):
    captured = {}

    def fake_fetch_json(url, *, namespace, key, params=None, max_age_seconds=None):
        captured["params"] = params
        return {
            "dates": [
                {"games": [{"gamePk": 1}, {"gamePk": 2}]},
                {"games": [{"gamePk": 3}]},
            ]
        }

    monkeypatch.setattr(api, "fetch_json", fake_fetch_json)
    games = api.schedule("2024-07-01", "2024-07-02")
    assert [g["gamePk"] for g in games] == [1, 2, 3]
    assert captured["params"]["startDate"] == "2024-07-01"
    assert captured["params"]["endDate"] == "2024-07-02"


def test_game_feed_refetches_when_final(monkeypatch):
    calls: list[float | None] = []

    def fake_fetch_json(url, *, namespace, key, params=None, max_age_seconds=None):
        calls.append(max_age_seconds)
        return {"gameData": {"status": {"abstractGameState": "Final"}}}

    monkeypatch.setattr(api, "fetch_json", fake_fetch_json)
    api.game_feed(744914)
    # first call with TTL, second call with no expiry once we learn it is Final
    assert calls == [api._LIVE_TTL, None]


def test_game_feed_single_fetch_when_assumed_final(monkeypatch):
    calls: list[float | None] = []

    def fake_fetch_json(url, *, namespace, key, params=None, max_age_seconds=None):
        calls.append(max_age_seconds)
        return {"gameData": {"status": {"abstractGameState": "Final"}}}

    monkeypatch.setattr(api, "fetch_json", fake_fetch_json)
    api.game_feed(744914, assume_final=True)
    assert calls == [None]
