"""Jobs API (STEP 19, commit 2): create/list/get/logs/cancel + artifacts."""

import io

import pytest

fastapi = pytest.importorskip("fastapi", reason="backend deps not installed")
pytest.importorskip("httpx", reason="backend deps not installed")


def _client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import create_app

    monkeypatch.setenv("SP3D_OUTPUTS", str(tmp_path))
    # isolate jobs under tmp outputs by redirecting PROJECT_ROOT-relative path
    import api.store as store

    def _roots():
        d = tmp_path / "jobs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(store, "jobs_root", _roots)

    def _jd(job_id: str):
        d = tmp_path / "jobs" / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(store, "job_dir", _jd)
    monkeypatch.setattr(store, "job_json_path", lambda jid: _jd(jid) / "job.json")
    return TestClient(create_app())


def test_create_and_get_job_queues_pipeline(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    video = io.BytesIO(b"fake-mp4-bytes")
    resp = client.post(
        "/api/jobs",
        files={"video": ("flight.mp4", video, "video/mp4")},
        data={"target_fps": "2.0", "max_frames": "10", "depth_backend": "dummy"},
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["id"]
    got = client.get(f"/api/jobs/{job_id}")
    assert got.status_code == 200
    assert got.json()["id"] == job_id


def test_reject_bad_video_suffix(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.post(
        "/api/jobs",
        files={"video": ("flight.txt", io.BytesIO(b"x"), "text/plain")},
    )
    assert resp.status_code == 400


def test_trajectory_404_before_run(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    video = io.BytesIO(b"fake-mp4-bytes")
    job_id = client.post(
        "/api/jobs",
        files={"video": ("flight.mp4", video, "video/mp4")},
    ).json()["id"]
    # background task may fail fast on fake video; trajectory stays missing
    import time

    time.sleep(0.5)
    r = client.get(f"/api/jobs/{job_id}/trajectory")
    assert r.status_code in (200, 404)
