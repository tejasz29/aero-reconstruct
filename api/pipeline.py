"""STEP 19 — offline pipeline runner (Steps 2-10) for one job.

All heavy ``src.*`` imports happen inside stage functions so importing
this module (and ``api.main``) never requires torch/open3d/COLMAP.
Each stage updates ``job.json`` + ``run.log`` so the React UI can poll.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import store
from .schemas import STAGES


def _mark(job_id: str, stage: str, status: str, message: str = "") -> None:
    try:
        job = store.load_job(job_id)
    except FileNotFoundError:
        return
    for s in job.stages:
        if s.name == stage:
            s.status = status
            s.message = message
    if status == "running":
        job.status = "RUNNING"
    store.save_job(job)
    store.append_log(job_id, f"[{stage}] {status} {message}".strip())


def _warn(job_id: str, message: str) -> None:
    try:
        job = store.load_job(job_id)
    except FileNotFoundError:
        return
    job.warnings.append(message)
    store.save_job(job)
    store.append_log(job_id, f"[warn] {message}")


def _job_paths(job_id: str) -> dict[str, Path]:
    root = store.job_dir(job_id)
    return {
        "root": root,
        "frames": root / "frames",
        "trajectory": root / "trajectory",
        "georef": root / "georef",
        "depth": root / "depth",
        "pointcloud": root / "pointcloud",
        "reports": root / "reports",
        "uploads": root / "uploads",
    }


def run_job(job_id: str) -> None:
    """Run Steps 2-10 sequentially; never raises (records FAILED instead)."""
    from src.common.config_loader import load_config

    paths = _job_paths(job_id)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    try:
        job = store.load_job(job_id)
    except FileNotFoundError:
        return
    params: dict[str, Any] = dict(job.params or {})
    store.append_log(job_id, f"job {job_id} started params={params}")

    try:
        cfg = load_config(params.get("config")) if params.get("config") else load_config()
    except Exception as exc:  # noqa: BLE001 — config errors become job failure
        _fail(job_id, f"config load failed: {exc}")
        return

    video = params.get("video_path")
    if not video:
        _fail(job_id, "no video_path uploaded (POST /api/jobs needs a video file)")
        return

    # STEP 2 — extract frames
    _mark(job_id, "extract-frames", "running")
    try:
        from src.video.frame_extractor import extract_frames

        extract_frames(
            video,
            paths["frames"],
            target_fps=float(params.get("target_fps", 2.0)),
            max_frames=int(params.get("max_frames", 2000)),
            max_width=int(params.get("max_width", 1920)),
        )
        _mark(job_id, "extract-frames", "done")
    except Exception as exc:  # noqa: BLE001
        _mark(job_id, "extract-frames", "failed", str(exc))
        _fail(job_id, f"extract-frames: {exc}")
        return

    # STEP 3 — keyframes
    _mark(job_id, "select-keyframes", "running")
    try:
        from src.common.config_loader import get
        from src.video.keyframes import run_selection

        run_selection(
            paths["frames"],
            blur_threshold=float(get(cfg, "video.quality.blur_threshold", 100.0)),
            exposure_min=float(get(cfg, "video.quality.min_exposure_mean", 15.0)),
            exposure_max=float(get(cfg, "video.quality.max_exposure_mean", 240.0)),
            min_features=int(get(cfg, "video.quality.min_features", 300)),
            min_time_gap_s=float(get(cfg, "video.keyframes.min_time_gap_s", 0.0)),
            dedup_hamming_threshold=int(
                get(cfg, "video.keyframes.dedup_hamming_threshold", 5)),
            max_keep=int(get(cfg, "video.keyframes.max_keep", 600)),
            detector=str(get(cfg, "video.quality.detector", "orb")),
        )
        _mark(job_id, "select-keyframes", "done")
    except Exception as exc:  # noqa: BLE001
        _mark(job_id, "select-keyframes", "failed", str(exc))
        _fail(job_id, f"select-keyframes: {exc}")
        return

    # STEP 4 — calibration (provided/checkerboard/charuco via config)
    _mark(job_id, "calibrate", "running")
    try:
        from src.calibration.runner import resolve_calibration

        resolve_calibration(cfg)
        _mark(job_id, "calibrate", "done")
    except Exception as exc:  # noqa: BLE001
        _mark(job_id, "calibrate", "failed", str(exc))
        _fail(job_id, f"calibrate: {exc}")
        return

    # STEP 5 — poses
    _mark(job_id, "reconstruct-poses", "running")
    try:
        from src.sfm.runner import run_reconstruction

        run_reconstruction(cfg, frames_dir=paths["frames"],
                           output_dir=paths["trajectory"])
        _mark(job_id, "reconstruct-poses", "done")
    except Exception as exc:  # noqa: BLE001
        _mark(job_id, "reconstruct-poses", "failed", str(exc))
        _fail(job_id, f"reconstruct-poses: {exc}")
        return

    # STEP 6 — trajectory viz (never fatal: PNGs only)
    _mark(job_id, "show-trajectory", "running")
    try:
        from src.sfm.visualize import plot_trajectory, read_poses_csv

        poses_csv = paths["trajectory"] / "camera_poses.csv"
        plot_trajectory(read_poses_csv(poses_csv), paths["reports"], cfg)
        _mark(job_id, "show-trajectory", "done")
    except Exception as exc:  # noqa: BLE001
        _warn(job_id, f"show-trajectory skipped: {exc}")
        _mark(job_id, "show-trajectory", "done", f"skipped: {exc}")

    # STEP 7 — GPS (optional: skip cleanly when no log uploaded)
    gps_file = params.get("gps_path")
    _mark(job_id, "convert-gps", "running")
    if not gps_file:
        _warn(job_id, "no GPS log uploaded — metric alignment will use scale=1")
        _mark(job_id, "convert-gps", "done", "skipped (no gps.csv)")
    else:
        try:
            from pathlib import Path as _P

            from src.georef.gps import run_gps_conversion

            run_gps_conversion(cfg, gps_file=_P(gps_file),
                               output_dir=paths["georef"])
            _mark(job_id, "convert-gps", "done")
        except Exception as exc:  # noqa: BLE001
            _warn(job_id, f"convert-gps skipped: {exc}")
            _mark(job_id, "convert-gps", "done", f"skipped: {exc}")

    # STEP 8 — alignment (rejection is a warning, not fatal: STEP 10 uses s=1)
    _mark(job_id, "align-trajectory", "running")
    try:
        from src.georef.runner import run_alignment

        result = run_alignment(
            cfg,
            poses_csv=paths["trajectory"] / "camera_poses.csv",
            gps_csv=paths["georef"] / "gps_metric.csv",
            output_dir=paths["georef"],
        )
        if not result.success:
            _warn(job_id, f"alignment REJECTED ({result.reject_reason}) — "
                          "cloud stays metric-via-scale s=1, NOT absolute CRS")
        _mark(job_id, "align-trajectory", "done")
    except Exception as exc:  # noqa: BLE001
        _warn(job_id, f"align-trajectory skipped: {exc}")
        _mark(job_id, "align-trajectory", "done", f"skipped: {exc}")

    # STEP 9 — depth (dummy CPU by default; HF fallback inside runner)
    _mark(job_id, "predict-depth", "running")
    try:
        from src.depth.runner import run_depth_prediction

        run_depth_prediction(cfg, frames_dir=paths["frames"],
                             output_dir=paths["depth"],
                             backend=params.get("depth_backend"),
                             device=params.get("device"))
        _mark(job_id, "predict-depth", "done")
    except Exception as exc:  # noqa: BLE001
        _mark(job_id, "predict-depth", "failed", str(exc))
        _fail(job_id, f"predict-depth: {exc}")
        return

    # STEP 10 — unproject to colored clouds
    _mark(job_id, "unproject-depth", "running")
    try:
        from src.fusion.runner import run_unprojection

        res = run_unprojection(
            cfg,
            depth_index=paths["depth"] / "depth_index.csv",
            poses_csv=paths["trajectory"] / "camera_poses.csv",
            frames_dir=paths["frames"],
            output_dir=paths["pointcloud"],
            stride=params.get("stride"),
        )
        job = store.load_job(job_id)
        job.outputs = {
            "scene_ply": str(res.get("scene_ply", "")),
            "n_points": res.get("n_points", 0),
            "scale": res.get("scale", 1.0),
        }
        store.save_job(job)
        _mark(job_id, "unproject-depth", "done")
    except Exception as exc:  # noqa: BLE001
        _mark(job_id, "unproject-depth", "failed", str(exc))
        _fail(job_id, f"unproject-depth: {exc}")
        return

    # STEP 11 — fuse frame clouds (non-fatal: raw STEP 10 scene stays valid)
    _mark(job_id, "fuse-cloud", "running")
    try:
        from src.fusion.runner import run_fusion

        fused = run_fusion(
            cfg,
            cloud_index=paths["pointcloud"] / "cloud_index.csv",
            depth_index=paths["depth"] / "depth_index.csv",
            output_dir=paths["pointcloud"],
        )
        job = store.load_job(job_id)
        job.outputs["fused_ply"] = fused.get("fused_ply", "")
        job.outputs["fused_points"] = fused.get("n_points", 0)
        store.save_job(job)
        _mark(job_id, "fuse-cloud", "done")
    except Exception as exc:  # noqa: BLE001
        _warn(job_id, f"fuse-cloud skipped: {exc}")
        _mark(job_id, "fuse-cloud", "done", f"skipped: {exc}")

    # Finish
    job = store.load_job(job_id)
    for s in job.stages:
        if s.status == "pending":
            s.status = "skipped"
    job.status = "COMPLETED_WITH_WARNINGS" if job.warnings else "COMPLETED"
    store.save_job(job)
    store.append_log(job_id, f"job {job_id} finished: {job.status}")


def _fail(job_id: str, message: str) -> None:
    try:
        job = store.load_job(job_id)
    except FileNotFoundError:
        return
    job.status = "FAILED"
    job.error = message
    store.save_job(job)
    store.append_log(job_id, f"[failed] {message}")


def stage_names() -> list[str]:
    return list(STAGES)
