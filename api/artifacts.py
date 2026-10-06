"""STEP 19 — read-only artifact endpoints over Steps 5-10 outputs.

Serves parsed trajectory / point-cloud summaries plus raw files under
``outputs/jobs/{id}/`` with a fallback to the global ``outputs/`` demo
run, so the viewer works before any job is queued.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.common.paths import PROJECT_ROOT

from . import store

router = APIRouter(prefix="/api", tags=["artifacts"])


def _roots(job_id: str) -> list[Path]:
    job_root = PROJECT_ROOT / "outputs" / "jobs" / job_id
    return [job_root, PROJECT_ROOT / "outputs"]


def _find(job_id: str, *rel: str) -> Path | None:
    for root in _roots(job_id):
        p = root.joinpath(*rel)
        if p.is_file():
            return p
    return None


def _list(job_id: str, *rel: str, pattern: str = "*") -> list[Path]:
    seen: list[Path] = []
    for root in _roots(job_id):
        d = root.joinpath(*rel)
        if d.is_dir():
            seen.extend(sorted(d.glob(pattern)))
    # de-dupe by filename, job-local wins
    by_name: dict[str, Path] = {}
    for p in seen:
        by_name.setdefault(p.name, p)
    job_first = [p for p in seen if "jobs" in p.parts]
    rest = [p for p in seen if "jobs" not in p.parts and p.name not in
            {q.name for q in job_first}]
    return job_first + rest


@router.get("/jobs/{job_id}/trajectory", response_model=dict)
def trajectory(job_id: str) -> dict[str, Any]:
    poses_csv = _find(job_id, "trajectory", "camera_poses.csv")
    if poses_csv is None:
        raise HTTPException(status_code=404, detail="camera_poses.csv not ready")
    poses: list[dict[str, Any]] = []
    kept = rejected = 0
    with open(poses_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                kept_flag = int(float(row.get("kept", 0)))
            except ValueError:
                kept_flag = 0
            if kept_flag:
                kept += 1
            else:
                rejected += 1
            try:
                poses.append({
                    "frame_id": int(float(row.get("frame_id", 0))),
                    "timestamp_s": float(row.get("timestamp_s", 0.0)),
                    "kept": bool(kept_flag),
                    "reject_reason": row.get("reject_reason", ""),
                    "t": [float(row.get("tx", 0.0)), float(row.get("ty", 0.0)),
                          float(row.get("tz", 0.0))],
                })
            except ValueError:
                continue
    return {"poses_csv": str(poses_csv), "kept": kept, "rejected": rejected,
            "poses": poses}


@router.get("/jobs/{job_id}/pointcloud", response_model=dict)
def pointcloud(job_id: str) -> dict[str, Any]:
    scene = _find(job_id, "pointcloud", "scene.ply")
    index = _find(job_id, "pointcloud", "cloud_index.csv")
    report_p = _find(job_id, "reports", "unproject_report.json")
    report: dict[str, Any] = {}
    if report_p is not None:
        try:
            report = json.loads(report_p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
    frames: list[dict[str, Any]] = []
    if index is not None:
        with open(index, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                frames.append(dict(row))
    if scene is None:
        raise HTTPException(status_code=404, detail="scene.ply not ready")
    return {
        "scene_ply": str(scene),
        "scene_url": f"/api/files/{job_id}/pointcloud/scene.ply",
        "n_points": report.get("n_points", len(frames)),
        "scale": report.get("scale", 1.0),
        "metric_via_gps_scale": report.get("metric_via_gps_scale", True),
        "absolute_crs": report.get("absolute_crs", False),
        "frames": frames,
        "report": report,
    }


@router.get("/jobs/{job_id}/reports", response_model=dict)
def reports(job_id: str) -> dict[str, Any]:
    names = ["calibration_report.json", "trajectory_report.json",
             "gps_report.json", "aligned_trajectory_report.json",
             "aligned_trajectory_transform.json", "depth_report.json",
             "unproject_report.json"]
    out: dict[str, Any] = {}
    for name in names:
        p = _find(job_id, "reports", name)
        if p is None:
            continue
        try:
            out[name] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out[name] = {"_error": f"unreadable {p}"}
    return {"reports": out}


@router.get("/files/{job_id}/{subdir}/{filename}")
def files(job_id: str, subdir: str, filename: str):
    if ".." in subdir or ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="bad path")
    if subdir not in ("pointcloud", "trajectory", "reports", "depth",
                      "georef", "frames"):
        raise HTTPException(status_code=400, detail="bad subdir")
    p = _find(job_id, subdir, filename)
    if p is None:
        # also allow job-local uploads/frames listing via frames subdir
        raise HTTPException(status_code=404, detail="file not found")
    media = None
    if p.suffix == ".ply":
        media = "text/plain"
    return FileResponse(path=str(p), filename=p.name, media_type=media)
