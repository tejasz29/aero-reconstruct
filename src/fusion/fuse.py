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
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys, points = keys[order], np.asarray(points, dtype=np.float64)[order]
    colors = np.asarray(colors, dtype=np.float64)[order]
    conf = np.asarray(confidence, dtype=np.float64)[order].reshape(-1)
    uniq, starts = np.unique(keys, axis=0, return_index=True)
    # np.unique sorts; rebuild counts via searchsorted on sorted keys
    sorted_idx = np.argsort([tuple(k) for k in keys.tolist()], kind="stable")
    keys, points, colors, conf = keys[sorted_idx], points[sorted_idx], colors[sorted_idx], conf[sorted_idx]
    uniq, idx, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    out_pts, out_cols, out_conf = [], [], []
    for u, c in zip(counts, np.split(np.arange(len(keys)), np.cumsum(counts)[:-1])):
        w = conf[c]
        wsum = float(w.sum())
        if not np.isfinite(wsum) or wsum <= 0.0:
            w = np.ones_like(w)
            wsum = float(w.sum())
        w /= wsum
        out_pts.append((points[c] * w[:, None]).sum(axis=0))
        out_cols.append(np.clip((colors[c] * w[:, None]).sum(axis=0), 0, 255))
        out_conf.append(float(conf[c].mean()))
    return (np.asarray(out_pts), np.asarray(out_cols, dtype=np.float64),
            np.asarray(out_conf, dtype=np.float64))
