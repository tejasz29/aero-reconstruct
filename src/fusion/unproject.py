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
