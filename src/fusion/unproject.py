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


def valid_mask(depth: np.ndarray, min_depth: float = 1e-6,
               max_depth: float = 1e9) -> np.ndarray:
    """Keep mask for finite depth in (min, max]; never silently zero."""
    if not min_depth >= 0.0:
        raise ValueError(f"min_depth must be >= 0, got {min_depth}")
    if not max_depth > min_depth:
        raise ValueError("max_depth must exceed min_depth")
    d = np.asarray(depth)
    return np.isfinite(d) & (d > min_depth) & (d <= max_depth)


def colored_cloud(depth: np.ndarray, rgb: np.ndarray, fx: float, fy: float,
                  cx: float, cy: float, R_wc: np.ndarray, C: np.ndarray,
                  scale: float = 1.0, confidence: np.ndarray | None = None,
                  stride: int = 1, min_depth: float = 1e-6,
                  max_depth: float = 1e9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth+RGB -> (points Nx3 metric-via-scale, colors Nx3 uint8, conf N).

    Drops invalid pixels via :func:`valid_mask`; ``stride`` thins pixels
    (2 = every other row/col) from config to bound cloud size.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    d = np.asarray(depth)
    img = np.asarray(rgb)
    if d.shape != img.shape[:2]:
        raise ValueError(f"depth {d.shape} != image {img.shape[:2]}")
    if img.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got shape {img.shape}")
    mask = valid_mask(d, min_depth, max_depth)
    if stride > 1:
        keep = np.zeros_like(mask, dtype=bool)
        keep[::stride, ::stride] = True
        mask = mask & keep
    cam = unproject_depth(np.where(mask, d, np.nan), fx, fy, cx, cy)
    pts_cam = cam[mask]
    world = apply_pose(pts_cam, R_wc, C)
    metric = apply_scale(world, scale)
    colors = img[mask].astype(np.uint8)
    if confidence is None:
        conf = np.ones(len(metric), dtype=np.float32)
    else:
        conf = np.asarray(confidence, dtype=np.float32)[mask]
    return metric, colors, conf
