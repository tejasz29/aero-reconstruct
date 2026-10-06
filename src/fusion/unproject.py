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
