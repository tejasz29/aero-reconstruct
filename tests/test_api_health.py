"""Backend health (STEP 19, commit 1): /api/health is import-light."""

import pytest

fastapi = pytest.importorskip("fastapi", reason="backend deps not installed")
httpx = pytest.importorskip("httpx", reason="backend deps not installed")

from api.main import create_app, pipeline_progress  # noqa: E402


def test_pipeline_progress_reports_steps_1_to_10():
    prog = pipeline_progress()
    assert prog["steps_done"] == list(range(1, 11))
    assert prog["next_step"] == 11


def test_health_endpoint_ok():
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["next_step"] == 11
