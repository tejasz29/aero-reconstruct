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

    def validate(self) -> "Intrinsics":
        """Sanity-check physical plausibility; raises ValueError when broken."""
        if not (np.isfinite(self.fx) and self.fx > MIN_FOCAL_PX):
            raise ValueError(f"fx must be > {MIN_FOCAL_PX}, got {self.fx}")
        if not (np.isfinite(self.fy) and self.fy > MIN_FOCAL_PX):
            raise ValueError(f"fy must be > {MIN_FOCAL_PX}, got {self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"image size must be positive, got {(self.width, self.height)}")
        if not all(np.isfinite(d) for d in self.distortion):
            raise ValueError("distortion coefficients must be finite")
        if len(self.distortion) != DISTORTION_SIZE:
            raise ValueError(
                f"distortion must have {DISTORTION_SIZE} coeffs "
                f"(k1,k2,p1,p2,k3), got {len(self.distortion)}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "image_width": self.width,
            "image_height": self.height,
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "distortion": list(self.distortion),
            "reprojection_error_px": (float(self.reprojection_error_px)
                                      if self.reprojection_error_px is not None else None),
            "calibrated_on": self.calibrated_on,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Intrinsics":
        cam = data.get("camera") or data
        dist = cam.get("distortion") or []
        return cls(
            fx=float(cam["fx"]),
            fy=float(cam["fy"]),
            cx=float(cam["cx"]),
            cy=float(cam["cy"]),
            width=int(cam["image_width"]),
            height=int(cam["image_height"]),
            distortion=tuple(float(d) for d in dist[:DISTORTION_SIZE]) if dist else (0.0,) * DISTORTION_SIZE,
            source=str(cam.get("source") or "unknown"),
            reprojection_error_px=cam.get("reprojection_error_px"),
            calibrated_on=cam.get("calibrated_on"),
        )