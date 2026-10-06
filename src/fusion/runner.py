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
