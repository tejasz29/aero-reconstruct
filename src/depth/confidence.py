"""STEP 9 — confidence maps for monocular depth.

Heuristic, not learned: flat/textureless areas and strong depth edges are
less trustworthy. Confidence stays in [0, 1] and is preserved end-to-end
so STEP 11 fusion can weight points.
"""

from __future__ import annotations

import numpy as np


def gradient_magnitude(depth: np.ndarray) -> np.ndarray:
    """Central-difference |grad| of a HxW depth map (float64)."""
    d = np.asarray(depth, dtype=np.float64)
    gy, gx = np.gradient(d)
    return np.sqrt(gx * gx + gy * gy)


def confidence_from_depth(depth: np.ndarray) -> np.ndarray:
    """Map relative depth -> confidence in [0, 1].

    High local variation (edges, spikes) lowers confidence via
    ``1 / (1 + |grad| / scale)`` where scale is the median gradient
    (robust to outliers). Flat maps yield ~1 everywhere.
    """
    d = np.asarray(depth, dtype=np.float64)
    grad = gradient_magnitude(d)
    scale = float(np.median(grad)) + 1e-6
    conf = 1.0 / (1.0 + grad / scale)
    return np.clip(conf, 0.0, 1.0).astype(np.float32)
