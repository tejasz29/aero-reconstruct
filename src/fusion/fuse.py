"""STEP 11 — point-cloud fusion (confidence weighting, voxel downsample).

Merges STEP 10 per-frame clouds into one deduplicated scene. Pure-numpy
baseline; open3d is an optional fast path only. Metric-via-GPS-scale is
preserved; absolute CRS stays deferred to STEP 16.
"""

from __future__ import annotations

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.fusion")


def voxel_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Quantize Nx3 points to int64 voxel indices (floor-divide)."""
    if voxel_size <= 0.0 or not np.isfinite(voxel_size):
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if not bool(np.all(np.isfinite(pts))):
        raise ValueError("points must be finite for voxel hashing")
    return np.floor(pts / float(voxel_size)).astype(np.int64)
