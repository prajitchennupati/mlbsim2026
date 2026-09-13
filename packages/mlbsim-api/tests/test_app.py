from __future__ import annotations

from fastapi.testclient import TestClient

from mlbsim_api.cache import Cache
from mlbsim_api.main import create_app

client = TestClient(create_app())

_EXPECTED_PATHS = {
    "/api/v1/games",
    "/api/v1/games/{game_pk}",
    "/api/v1/games/{game_pk}/innings",
    "/api/v1/games/{game_pk}/players",
    "/api/v1/games/{game_pk}/simulation",
    "/api/v1/games/{game_pk}/live",
    "/api/v1/teams",
    "/api/v1/teams/{abbr}",
    "/api/v1/teams/{abbr}/schedule",
    "/api/v1/standings",
    "/api/v1/playoffs",
    "/api/v1/simulations/season/latest",
    "/api/v1/models",
    "/api/v1/models/{model_id}/evaluation",
    "/api/v1/predictions/history",
    "/api/v1/predictions/summary",
    "/api/v1/predictions/{pred_id}/explanation",
    "/healthz",
    "/version",
}


def test_all_v1_routes_are_mounted():
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths >= _EXPECTED_PATHS


def test_live_endpoint_is_a_placeholder_404():
    r = client.get("/api/v1/games/12345/live")
    assert r.status_code == 404
    assert "M9" in r.json()["detail"]


def test_cache_falls_back_to_local_without_redis():
    c = Cache()  # redis not running in CI -> local LRU
    c.set("k", {"v": 1}, ttl=5)
    assert c.get("k") == {"v": 1}
    assert c.get("missing") is None


def test_openapi_metadata():
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == "mlbplayoffs2026 API"
    assert "games" in {t["name"] for t in spec.get("tags", [])} or any(
        "games" in str(op.get("tags", []))
        for path in spec["paths"].values()
        for op in path.values()
    )
