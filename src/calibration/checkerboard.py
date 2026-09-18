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