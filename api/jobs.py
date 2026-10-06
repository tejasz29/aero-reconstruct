"""STEP 19 — job endpoints: upload video/GPS, queue Steps 2-10, poll status."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from . import store
from .pipeline import run_job
from .schemas import Job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}


@router.post("", response_model=dict)
def create_job(
    background: BackgroundTasks,
    video: UploadFile = File(...),
    gps: UploadFile | None = File(default=None),
    target_fps: float = Form(default=2.0),
    max_frames: int = Form(default=200),
    stride: int = Form(default=2),
    depth_backend: str = Form(default="dummy"),
) -> dict[str, Any]:
    suffix = Path(video.filename or "flight.mp4").suffix.lower() or ".mp4"
    if suffix not in _VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"unsupported video type: {suffix}")
    if depth_backend not in ("dummy", "hf", "auto"):
        raise HTTPException(status_code=400, detail="depth_backend must be dummy|hf|auto")

    job_id = uuid.uuid4().hex[:12]
    job = Job.new(job_id, params={
        "target_fps": target_fps,
        "max_frames": max_frames,
        "stride": stride,
        "depth_backend": depth_backend,
    })
    store.save_job(job)

    uploads = store.job_dir(job_id) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    video_path = uploads / f"flight{suffix}"
    with open(video_path, "wb") as fh:
        shutil.copyfileobj(video.file, fh)
    gps_path = None
    if gps is not None and gps.filename:
        gps_path = uploads / "gps.csv"
        with open(gps_path, "wb") as fh:
            shutil.copyfileobj(gps.file, fh)

    job.params["video_path"] = str(video_path)
    if gps_path is not None:
        job.params["gps_path"] = str(gps_path)
    store.save_job(job)
    store.append_log(job_id, f"uploaded {video.filename} -> {video_path}")

    background.add_task(run_job, job_id)
    return {"id": job_id, "status": job.status}


@router.get("", response_model=dict)
def list_jobs() -> dict[str, Any]:
    return {"jobs": [j.to_dict() for j in store.list_jobs()]}


@router.get("/{job_id}", response_model=dict)
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return store.load_job(job_id).to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="job not found")


@router.get("/{job_id}/logs", response_model=dict)
def get_logs(job_id: str, tail: int = 200) -> dict[str, Any]:
    p = store.job_dir(job_id) / "run.log"
    # job_dir() creates the dir; guard jobs that never existed via job.json
    if not (store.jobs_root() / job_id / "job.json").is_file():
        raise HTTPException(status_code=404, detail="job not found")
    if not p.is_file():
        return {"logs": []}
    lines = p.read_text(encoding="utf-8").splitlines()[-max(1, tail):]
    return {"logs": lines}


@router.post("/{job_id}/cancel", response_model=dict)
def cancel_job(job_id: str) -> dict[str, Any]:
    try:
        job = store.load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS", "FAILED"):
        return job.to_dict()
    job.status = "CANCELLED"
    store.save_job(job)
    store.append_log(job_id, "cancel requested by user")
    return job.to_dict()
