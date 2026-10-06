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
