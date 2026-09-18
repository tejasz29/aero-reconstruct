"""STEP 4 — checkerboard calibration.

Each photo of a known checkerboard gives a set of grid points whose physical
spacing is known (``square_size_m``). Detected corners + those object points
are fed to ``cv2.calibrateCamera``, which solves for ``K`` and the distortion
coefficients by least squares. The returned RMS reprojection error tells us
how trustworthy the model is (~<1 px is good).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.calibration.intrinsics import Intrinsics
from src.common.logging_utils import get_logger

log = get_logger("sp3d.calibration")


@dataclass
class CalibrationResult:
    intrinsics: Intrinsics
    rms_px: float
    used_images: list[Path]
    rejected_images: list[tuple[Path, str]]
    per_view_rms_px: list[tuple[Path, float, int]] = field(default_factory=list)
    method: str = "checkerboard"
    pattern_size: tuple[int, int] | None = None


def checkerboard_object_points(pattern_size: tuple[int, int],
                               square_size_m: float) -> np.ndarray:
    """Object points for one view: z=0 grid, ordered col-major like OpenCV."""
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_size_m)
    return objp


def detect_checkerboard(image: np.ndarray,
                        pattern_size: tuple[int, int]) -> np.ndarray | None:
    """Inner-corner positions as float32 Nx2, or None when not detected.

    Uses the robust ``findChessboardCornersSB`` detector when available
    (OpenCV >= 4.5.1) and falls back to the classic detector + corner
    refinement otherwise.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "findChessboardCornersSB"):
        # SB detector only accepts CALIB_CB_NORMALIZE_IMAGE / FAST_CHECK flags.
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size)
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if ok:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2).astype(np.float32) if ok else None