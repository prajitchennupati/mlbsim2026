from __future__ import annotations

import datetime as dt

from mlbsim_features.asof import asof_join, asof_lookup, before, snapshot_hash


def _rows():
    return [
        {"team": 1, "game_date": "2024-04-01", "rf": 5},
        {"team": 1, "game_date": "2024-04-03", "rf": 2},
        {"team": 1, "game_date": "2024-04-07", "rf": 9},
        {"team": 2, "game_date": "2024-04-02", "rf": 1},
    ]


def test_before_is_strict_and_sorted():
    kept = before(_rows(), "2024-04-07")
    assert [r["game_date"] for r in kept] == ["2024-04-01", "2024-04-02", "2024-04-03"]
    # exact-match date is excluded (strict <)
    assert all(r["game_date"] != "2024-04-07" for r in kept)


def test_asof_lookup_returns_latest_prior():
    rows = [r for r in _rows() if r["team"] == 1]
    assert asof_lookup(rows, "2024-04-05")["rf"] == 2
    assert asof_lookup(rows, "2024-04-01") is None  # nothing strictly before


def test_asof_join_matches_by_key():
    queries = [
        {"team": 1, "as_of": "2024-04-04"},
        {"team": 2, "as_of": "2024-04-04"},
        {"team": 3, "as_of": "2024-04-04"},  # no events
    ]
    joined = {q["team"]: q for q in asof_join(_rows(), queries, by=["team"])}
    assert joined[1]["rf_asof"] == 2
    assert joined[2]["rf_asof"] == 1
    assert "rf_asof" not in joined[3]


def test_leakage_property_future_rows_do_not_change_the_past():
    """A lookup 'as of D' is invariant to rows dated on/after D being added later."""
    rows = _rows()
    as_of = "2024-04-05"
    baseline = asof_lookup([r for r in rows if r["team"] == 1], as_of)
    future = [
        *rows,
        {"team": 1, "game_date": "2024-04-05", "rf": 99},  # exactly D
        {"team": 1, "game_date": "2024-04-20", "rf": 99},  # after D
    ]
    assert asof_lookup([r for r in future if r["team"] == 1], as_of) == baseline


def test_snapshot_hash_is_order_independent_and_deterministic():
    a = snapshot_hash([3, 1, 2], max_source_ts=dt.date(2024, 4, 1))
    b = snapshot_hash([2, 3, 1], max_source_ts="2024-04-01")
    assert a == b
    assert snapshot_hash([1, 2, 3]) != snapshot_hash([1, 2, 4])
