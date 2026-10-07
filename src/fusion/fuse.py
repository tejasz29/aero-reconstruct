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


def _weighted_voxel_average(keys: np.ndarray, points: np.ndarray,
                            colors: np.ndarray,
                            confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse points sharing a voxel via confidence-weighted averages."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cols = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if not (len(keys) == len(pts) == len(cols) == len(conf)):
        raise ValueError("keys/points/colors/confidence length mismatch")
    uniq, _idx, inverse, counts = np.unique(keys, axis=0, return_index=True,
                                                return_inverse=True, return_counts=True)
    # np.unique with axis returns sorted uniq; group via inverse labels
    out_pts = np.zeros((len(uniq), 3))
    out_cols = np.zeros((len(uniq), 3))
    out_conf = np.zeros(len(uniq))
    for v in range(len(uniq)):
        mask = inverse == v
        w = np.where(np.isfinite(conf[mask]), conf[mask], 0.0)
        wsum = float(w.sum())
        if wsum <= 0.0:
            w = np.ones(mask.sum())
            wsum = float(w.sum())
        w = w / wsum
        out_pts[v] = (pts[mask] * w[:, None]).sum(axis=0)
        out_cols[v] = np.clip((cols[mask] * w[:, None]).sum(axis=0), 0, 255)
        out_conf[v] = float(conf[mask].mean()) if mask.sum() else 0.0
    return out_pts, out_cols, out_conf


def voxel_downsample(points: np.ndarray, colors: np.ndarray,
                     confidence: np.ndarray | None = None,
                     voxel_size: float = 0.10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample a cloud to one point per voxel (weighted average)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        raise ValueError("cannot downsample an empty cloud")
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(cols) != len(pts):
        raise ValueError(f"points {len(pts)} != colors {len(cols)}")
    conf = (np.ones(len(pts), dtype=np.float64) if confidence is None
            else np.asarray(confidence, dtype=np.float64).reshape(-1))
    if len(conf) != len(pts):
        raise ValueError("confidence length mismatch")
    keys = voxel_keys(pts, voxel_size)
    fused_pts, fused_cols, fused_conf = _weighted_voxel_average(keys, pts, cols, conf)
    return fused_pts, np.clip(fused_cols, 0, 255).astype(np.uint8), fused_conf
