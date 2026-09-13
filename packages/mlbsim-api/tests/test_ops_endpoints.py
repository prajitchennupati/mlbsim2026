from __future__ import annotations

from fastapi.testclient import TestClient

from mlbsim_api.main import create_app

client = TestClient(create_app())


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_version():
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "0.0.0"
    assert "environment" in body


def test_openapi_served():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert resp.json()["info"]["title"] == "mlbplayoffs2026 API"
