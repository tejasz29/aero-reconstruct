"""STEP 9 — depth inference (preprocess -> backend -> full-res depth+conf)."""

from __future__ import annotations

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.depth")


def ensure_finite(depth: np.ndarray) -> np.ndarray:
    """Replace non-finite pixels with the finite median (never NaN downstream)."""
    d = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(d)
    if bool(finite.all()):
        return d
    if not bool(finite.any()):
        raise ValueError("depth map has no finite pixels")
    fill = float(np.median(d[finite]))
    d[~finite] = fill
    return d
