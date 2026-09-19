"""STEP 5 — two-view geometry (classical SfM core).

Estimates camera motion from feature correspondences with OpenCV's RANSAC
essential-matrix recovery, validates every pair by triangulating inlier
correspondences (positive-depth cheirality + reprojection error) and by
measuring median parallax, then chains the relative motions into a single
world frame anchored at the first accepted keyframe.

OpenCV conventions used throughout:

* ``findEssentialMat``/``recoverPose`` work on *undistorted* pixel
  coordinates with the pinhole camera matrix ``K`` (distortion is removed
  from the raw feature points before estimation).
* A relative pose ``(R, t)`` maps camera-1 coordinates onto camera-2
  coordinates: ``X2 = R @ X1 + t``.
* A camera pose ``(R_wc, C)`` stores the world->camera rotation and the
  camera centre ``C`` in the world frame (world = first accepted keyframe).

Scale note: a monocular camera recovers motion up to ONE unknown global
scale (the essential matrix returns unit-length translation). STEP 8
resolves the absolute metric scale via visual<->GPS similarity alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.sfm")


@dataclass
class RelativePose:
    """Motion that takes camera-1 coordinates onto camera-2 coordinates."""

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inlier_count: int = 0
    mean_reproj_error_px: float = float("nan")
    median_parallax_px: float = 0.0
    valid: bool = False
    reject_reason: str = ""


@dataclass
class CameraPose:
    """One keyframe's pose in the trajectory world frame (== first frame)."""

    frame_id: int
    source_index: int
    timestamp_s: float
    filename: str
    kept: bool
    R_wc: np.ndarray | None = None     # world -> camera rotation (3x3)
    C: np.ndarray | None = None        # camera centre in world (3,)
    inliers: int = 0
    mean_reproj_error_px: float | None = None
    median_parallax_px: float | None = None
    reject_reason: str = ""


def _undistort_points(pts: np.ndarray, K: np.ndarray, dist: tuple) -> np.ndarray:
    """Rectify raw (distorted) pixels to pinhole pixels for geometry code."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    dist = np.asarray(dist, dtype=np.float64)
    if pts.shape[0] == 0 or not np.any(dist):
        return pts
    rect = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, None, K)
    return rect.reshape(-1, 2)