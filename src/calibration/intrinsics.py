"""STEP 4 — camera intrinsics model.

Central data type used by every later stage: the pinhole ``K`` matrix plus
OpenCV's standard distortion model (``k1, k2, p1, p2, k3``)::

    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

``Intrinsics`` persists to/from ``calibration/camera.yaml`` (see the
placeholder schema there). ``project`` / ``unproject_pixel`` implement the
pinhole model — projection uses OpenCV's ``projectPoints`` so distortion is
applied exactly as downstream OpenCV code expects; unprojection is the pure
pinhole inverse (STEP 10 builds the full depth->3D path on it).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import yaml

from src.common.logging_utils import get_logger

log = get_logger("sp3d.calibration")

DISTORTION_SIZE = 5  # k1, k2, p1, p2, k3
MIN_FOCAL_PX = 1.0


@dataclass
class Intrinsics:
    """Verified camera model for one sensor/lens (all values in pixels)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple[float, ...] = (0.0,) * DISTORTION_SIZE
    source: str = "unknown"          # provided | checkerboard | charuco
    reprojection_error_px: float | None = None
    calibrated_on: str | None = None

    def camera_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])