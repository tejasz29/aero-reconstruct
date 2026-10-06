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


def uniform_confidence(shape: tuple[int, ...], value: float = 1.0) -> np.ndarray:
    """Constant confidence (fallback when depth is degenerate)."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence value must be in [0,1], got {value}")
    return np.full(shape, value, dtype=np.float32)


def validate_confidence(conf: np.ndarray, shape: tuple[int, ...]) -> None:
    """Raise when confidence breaks the [0,1]/shape/finite contract."""
    c = np.asarray(conf)
    if c.shape != tuple(shape):
        raise ValueError(f"confidence shape {c.shape} != depth shape {tuple(shape)}")
    if not np.all(np.isfinite(c)):
        raise ValueError("confidence must be finite")
    if bool((c < 0.0).any()) or bool((c > 1.0).any()):
        raise ValueError("confidence must lie in [0, 1]")
