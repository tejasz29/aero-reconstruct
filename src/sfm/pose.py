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


def _triangulate(points1: np.ndarray, points2: np.ndarray,
                 K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Triangulate undisturbed pixel pairs into camera-1 frame 3D points.

    ``P1 = K[I|0]``, ``P2 = K[R|t]``; returns (N, 3) points in camera-1
    coordinates ('X1').
    """
    p1 = cv2.triangulatePoints(K @ np.hstack([np.eye(3), np.zeros((3, 1))]),
                               K @ np.hstack([R, t.reshape(3, 1)]),
                               points1.reshape(-1, 2).T, points2.reshape(-1, 2).T)
    p1 = p1[:3] / p1[3]
    return p1.T


def _validate_pair(r1i: np.ndarray, r2i: np.ndarray, K: np.ndarray,
                   R: np.ndarray, t: np.ndarray,
                   min_inliers: int) -> tuple | None:
    """Triangulate and score one (R, t) hypothesis.

    Returns ``(front_count, X1, i1, i2, mean_error, parallax)`` where inlier
    points triangulate in front of both cameras and reproject tightly, or
    ``None`` when fewer than ``min_inliers`` correspondences survive.
    """
    X1 = _triangulate(r1i, r2i, K, R, t)
    z2 = (R @ X1.T).T + t
    front = (X1[:, 2] > 1e-3) & (z2[:, 2] > 1e-3)
    if int(front.sum()) < min_inliers:
        return None
    X1, i1, i2 = X1[front], r1i[front], r2i[front]
    p1_re = K @ X1.T
    p1_rows = (p1_re[:2] / p1_re[2]).T
    X2 = X1 @ R.T + t
    p2_rows = (K @ X2.T)[:2] / (K @ X2.T)[2]
    errors = np.linalg.norm(p1_rows - i1, axis=1) \
        + np.linalg.norm(p2_rows.T - i2, axis=1)
    parallax = float(np.median(np.linalg.norm(i2 - i1, axis=1)))
    return int(len(X1)), X1, i1, i2, float(np.mean(errors)), parallax