"""STEP 10 — depth -> 3D unprojection (pinhole inverse + pose + GPS scale).

Per-pixel ``X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy`` in the camera frame, then into
the STEP 5 world via ``Xw = R_wc.T @ Xcam + C``, then metric via the STEP 8
scale ``s``. Color comes from the source keyframe; confidence rides along
for STEP 11 fusion weighting.
"""

from __future__ import annotations

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.fusion")


def pixel_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Meshgrid of pixel coords (u right, v down), shape HxW each."""
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {(width, height)}")
    us, vs = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    return us, vs


def unproject_depth(depth: np.ndarray, fx: float, fy: float,
                    cx: float, cy: float) -> np.ndarray:
    """Camera-frame points (HxWx3) from a relative depth map.

    Vectorized pinhole inverse; invalid pixels (non-finite, <= 0) become NaN
    here and are dropped later by :func:`valid_mask` — never silently zero.
    """
    if fx <= 0 or fy <= 0:
        raise ValueError(f"focal must be positive, got {(fx, fy)}")
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"depth must be HxW, got shape {d.shape}")
    h, w = d.shape
    us, vs = pixel_grid(w, h)
    x = (us - cx) * d / fx
    y = (vs - cy) * d / fy
    return np.stack([x, y, d], axis=-1)


def reproject_points(points_cam: np.ndarray, fx: float, fy: float,
                     cx: float, cy: float) -> np.ndarray:
    """Camera-frame Nx3 -> pixels Nx2 (round-trip check for tests)."""
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 2))
    z = pts[:, 2]
    if bool((z <= 0).any()):
        raise ValueError("cannot reproject points with depth <= 0")
    u = pts[:, 0] * fx / z + cx
    v = pts[:, 1] * fy / z + cy
    return np.stack([u, v], axis=-1)


def apply_pose(points_cam: np.ndarray, R_wc: np.ndarray,
               C: np.ndarray) -> np.ndarray:
    """Camera -> STEP 5 world: ``Xw = R_wc.T @ Xcam + C`` (Nx3).

    Row-vector form: ``Xw_row = Xcam_row @ R_wc + C`` (since
    ``(R.T @ x).T = x.T @ R``).
    """
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R_wc, dtype=np.float64)
    c = np.asarray(C, dtype=np.float64).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f"R_wc must be 3x3, got {R.shape}")
    if pts.shape[0] == 0:
        return pts
    return pts @ R + c


def apply_scale(points: np.ndarray, scale: float) -> np.ndarray:
    """Relative -> metric via the STEP 8 GPS scale ``s`` (rotation-free).

    Only the scale part of the similarity is applied here; the rotation +
    offset that place the scene in an absolute CRS belong to STEP 16.
    """
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive and finite, got {scale}")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return pts * float(scale)
