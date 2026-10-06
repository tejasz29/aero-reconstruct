"""STEP 10 — runner: depth + poses + scale -> per-frame + merged clouds.

Inputs:  ``outputs/depth/depth_index.csv`` (STEP 9) + images,
         ``calibration/camera.yaml`` (STEP 4),
         ``outputs/trajectory/camera_poses.csv`` (STEP 5),
         ``outputs/reports/aligned_trajectory_transform.json`` (STEP 8 scale).
Outputs: ``outputs/pointcloud/frame_*.ply`` + ``outputs/pointcloud/scene.ply``
         ``outputs/pointcloud/cloud_index.csv``
         ``outputs/reports/unproject_report.json``
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.logging_utils import get_logger

log = get_logger("sp3d.fusion")


@dataclass(frozen=True)
class UnprojectPolicy:
    """Every tunable the unprojection obeys, resolved from config."""

    stride: int = 2
    min_depth: float = 1e-6
    max_depth: float = 1e9
    save_per_frame: bool = True

    def as_dict(self) -> dict:
        return {"stride": self.stride, "min_depth": self.min_depth,
                "max_depth": self.max_depth,
                "save_per_frame": self.save_per_frame}


def resolve_policy(cfg: dict, stride: int | None = None) -> UnprojectPolicy:
    """Read the ``fusion.*`` unproject knobs into a policy."""
    from src.common.config_loader import get

    defaults = UnprojectPolicy()
    return UnprojectPolicy(
        stride=int(stride if stride is not None
                   else get(cfg, "fusion.unproject_stride", defaults.stride)),
        min_depth=float(get(cfg, "fusion.min_depth_m", defaults.min_depth)),
        max_depth=float(get(cfg, "fusion.max_depth_m", defaults.max_depth)),
        save_per_frame=bool(get(cfg, "fusion.save_per_frame",
                                defaults.save_per_frame)),
    )


@dataclass(frozen=True)
class UnprojectPaths:
    """Input and output locations of one unprojection run."""

    depth_index: object
    poses_csv: object
    camera_yaml: object
    transform_json: object
    frames_dir: object
    output_dir: object
    index_csv: object
    scene_ply: object
    report_json: object


def resolve_paths(cfg: dict, depth_index=None, poses_csv=None,
                  camera_yaml=None, transform_json=None, frames_dir=None,
                  output_dir=None) -> UnprojectPaths:
    """Resolve STEP 4/5/8/9 inputs + outputs/pointcloud + reports."""
    from pathlib import Path

    from src.common.config_loader import get
    from src.common.paths import PROJECT_ROOT

    def _abs(p) -> Path:
        c = Path(p)
        return c if c.is_absolute() else PROJECT_ROOT / c

    frames = Path(frames_dir) if frames_dir else _abs(
        get(cfg, "paths.frames", "data/frames"))
    traj = _abs(get(cfg, "paths.trajectory", "outputs/trajectory"))
    georef = _abs(get(cfg, "paths.georef", "outputs/georef"))
    reports = _abs(get(cfg, "paths.reports", "outputs/reports"))
    out = Path(output_dir) if output_dir else _abs("outputs/pointcloud")
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    depth_idx = Path(depth_index) if depth_index else _abs("outputs/depth") / "depth_index.csv"
    name = str(get(cfg, "alignment.output_name", "aligned_trajectory"))
    return UnprojectPaths(
        depth_index=depth_idx,
        poses_csv=Path(poses_csv) if poses_csv else traj / "camera_poses.csv",
        camera_yaml=Path(camera_yaml) if camera_yaml else _abs(
            get(cfg, "calibration.file", "calibration/camera.yaml")),
        transform_json=Path(transform_json) if transform_json else reports / f"{name}_transform.json",
        frames_dir=frames, output_dir=out,
        index_csv=out / "cloud_index.csv", scene_ply=out / "scene.ply",
        report_json=reports / "unproject_report.json")


def _load_scale(transform_json) -> float:
    """Read STEP 8 scale ``s``; missing file means relative (s=1) + warning."""
    import json
    from pathlib import Path

    from src.georef.align import transform_from_dict

    p = Path(transform_json)
    if not p.is_file():
        log.warning("transform %s missing — using scale=1 (RELATIVE cloud)", p)
        return 1.0
    payload = json.loads(p.read_text(encoding="utf-8"))
    try:
        transform = transform_from_dict(payload)
    except ValueError:
        if "scale" in payload:
            return float(payload["scale"])
        raise
    return float(transform.scale)


def _poses_by_filename(poses_csv) -> dict:
    """Accepted STEP 5 poses keyed by filename (rejected frames skipped)."""
    from src.sfm.visualize import read_poses_csv

    poses = read_poses_csv(poses_csv)
    table = {}
    for p in poses:
        if p.kept and p.R_wc is not None and p.C is not None:
            table[p.filename] = p
    if not table:
        raise ValueError(f"no accepted poses in {poses_csv}")
    return table


def run_unprojection(cfg: dict, depth_index=None, poses_csv=None,
                     camera_yaml=None, transform_json=None, frames_dir=None,
                     output_dir=None, stride: int | None = None) -> dict:
    """STEP 10 entry point: per-frame + merged colored clouds."""
    from pathlib import Path

    import numpy as np

    from src.calibration.intrinsics import load_intrinsics
    from src.depth.io import load_depth_npy, read_depth_index
    from src.depth.preprocess import load_image
    from src.fusion.io import (save_ply, validate_cloud, write_cloud_index,
                              write_unproject_report)
    from src.fusion.types import SCALE_NOTE
    from src.fusion.unproject import cloud_stats, colored_cloud

    paths = resolve_paths(cfg, depth_index, poses_csv, camera_yaml,
                          transform_json, frames_dir, output_dir)
    policy = resolve_policy(cfg, stride=stride)
    entries = read_depth_index(paths.depth_index)
    if not entries:
        raise ValueError(f"no depth rows in {paths.depth_index}")
    intrinsics = load_intrinsics(paths.camera_yaml)
    poses = _poses_by_filename(paths.poses_csv)
    scale = _load_scale(paths.transform_json)
    log.info("unprojecting %d depth maps (stride=%d, scale=%.4f)",
             len(entries), policy.stride, scale)

    out_dir = Path(paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict] = []
    per_frame: list[tuple[np.ndarray, np.ndarray]] = []
    for row in entries:
        filename = str(row["filename"])
        pose = poses.get(filename)
        if pose is None:
            log.warning("no accepted pose for %s — skipped", filename)
            continue
        depth = load_depth_npy(row["depth_path"])
        rgb = load_image(Path(paths.frames_dir) / filename)
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"depth {depth.shape} != image {rgb.shape[:2]} "
                             f"for {filename}")
        conf = None
        if row.get("confidence_path"):
            try:
                conf = np.load(str(row["confidence_path"]))
            except FileNotFoundError:
                conf = None
        points, colors, conf_v = colored_cloud(
            depth, rgb, intrinsics.fx, intrinsics.fy, intrinsics.cx,
            intrinsics.cy, pose.R_wc, pose.C, scale=scale,
            confidence=conf, stride=policy.stride,
            min_depth=policy.min_depth, max_depth=policy.max_depth)
        if len(points) == 0:
            log.warning("empty cloud for %s — skipped", filename)
            continue
        validate_cloud(points, colors)
        stem = Path(filename).stem
        ply_path = ""
        if policy.save_per_frame:
            ply_path = str(save_ply(points, colors, out_dir / f"{stem}.ply"))
        index_rows.append({"frame_id": row["frame_id"], "filename": filename,
                           "n_points": len(points), "ply_path": ply_path,
                           "mean_confidence": round(float(np.mean(conf_v)), 6)})
        per_frame.append((points, colors))
    if not per_frame:
        raise ValueError("no frames produced points — check poses/depth overlap")
    merged_pts = np.concatenate([p for p, _ in per_frame], axis=0)
    merged_cols = np.concatenate([c for _, c in per_frame], axis=0)
    scene_ply = str(save_ply(merged_pts, merged_cols, paths.scene_ply))
    index_csv = str(write_cloud_index(index_rows, paths.index_csv))
    stats = cloud_stats(merged_pts)
    log.info("merged %d points from %d frames -> %s", len(merged_pts),
             len(per_frame), scene_ply)
    return {"paths": paths, "policy": policy, "scale": scale,
            "rows": index_rows, "per_frame": per_frame,
            "intrinsics": intrinsics, "note": SCALE_NOTE,
            "scene_ply": scene_ply, "index_csv": index_csv, "stats": stats,
            "n_points": len(merged_pts), "n_frames": len(per_frame)}
