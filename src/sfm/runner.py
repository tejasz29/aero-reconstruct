"""STEP 5 — sequential monocular SfM runner.

Inputs:  ``keyframes.csv`` (STEP 3) + a calibrated camera (STEP 4).
Outputs: ``outputs/trajectory/camera_poses.csv`` and
``outputs/reports/trajectory_report.json``.

The tracker consumes keyframes in timestamp order. The first accepted
keyframe fixes the world frame; each following keyframe is matched against
the previous accepted one (an *anchor*), its relative pose is estimated
with the essential-matrix RANSAC path and chained. Frames that fail any
validation are flagged with an audit ``reject_reason`` and skipped — the
anchor role simply stays on the last good frame so the trajectory resumes
cleanly.

Backend: the bundled classical OpenCV tracker is always used for the
offline MVP. ``sfm.backend=colmap`` is accepted but falls back to this
tracker with a warning when the COLMAP binary is unavailable (a native
COLMAP driver lands in a later step).
"""

from __future__ import annotations

import csv
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.calibration.intrinsics import Intrinsics, load_intrinsics
from src.common.config_loader import get
from src.common.logging_utils import get_logger
from src.common.paths import PROJECT_ROOT
from src.sfm.pose import CameraPose

log = get_logger("sp3d.sfm")

KEYFRAMES_HEADER = ["frame_id", "source_index", "timestamp_s", "filename"]
POSES_HEADER = [
    "frame_id", "source_index", "timestamp_s", "filename",
    "kept", "reject_reason",
    "tx", "ty", "tz",
    "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22",
    "inliers", "mean_reproj_error_px", "median_parallax_px",
]


@dataclass
class TrajectoryResult:
    frames_dir: Path
    keyframes_path: Path
    poses_path: Path
    report_path: Path
    backend: str
    poses: list[CameraPose] = field(default_factory=list)

    @property
    def kept(self) -> list[CameraPose]:
        return [p for p in self.poses if p.kept]

    @property
    def rejected(self) -> list[CameraPose]:
        return [p for p in self.poses if not p.kept]

    @property
    def rejected_counts(self) -> dict[str, int]:
        return dict(Counter(p.reject_reason for p in self.rejected))

    @property
    def mean_reproj_error_px(self) -> float | None:
        errors = [p.mean_reproj_error_px for p in self.kept
                  if p.mean_reproj_error_px is not None]
        return float(np.mean(errors)) if errors else None


def _resolve_intrinsics(intrinsics: Intrinsics | str | Path | None,
                        cfg: dict) -> Intrinsics:
    """Accept an Intrinsics object or a path; default to ``calibration.file``."""
    if intrinsics is None:
        path = Path(get(cfg, "calibration.file", "calibration/camera.yaml"))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return load_intrinsics(path)
    if isinstance(intrinsics, (str, Path)):
        return load_intrinsics(intrinsics)
    return intrinsics.validate()


def _resolve_backend(cfg: dict) -> str:
    """Always run the bundled tracker for MVP; warn on a colmap request."""
    requested = str(get(cfg, "sfm.backend", "colmap")).lower()
    if requested == "colmap":
        if shutil.which("colmap") is None:
            log.warning("COLMAP binary not found — using the bundled OpenCV "
                        "tracker (sfm.backend=%s).", requested)
        else:
            log.info("COLMAP found on PATH; the bundled OpenCV tracker is "
                     "used by this MVP (a native COLMAP driver lands later).")
    return "opencv"