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
