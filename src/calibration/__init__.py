"""STEP 4 — camera intrinsics: loading + checkerboard/Charuco calibration.

Public surface:

* ``Intrinsics`` / ``load_intrinsics`` / ``save_intrinsics`` —
  the canonical camera model persisted to ``calibration/camera.yaml``.
* ``project`` / ``unproject_pixel`` / ``undistort_image`` — the pinhole +
  distortion model used by every downstream stage.
* ``calibrate_checkerboard`` / ``calibrate_charuco`` — board-based solvers.
* ``resolve_calibration`` — config-driven runner (provided | checkerboard |
  charuco) that writes the camera model + validation report.
"""

from src.calibration.charuco import calibrate_charuco
from src.calibration.checkerboard import (
    CalibrationResult,
    calibrate_checkerboard,
    detect_checkerboard,
)
from src.calibration.intrinsics import (
    Intrinsics,
    load_intrinsics,
    make_intrinsics,
    project,
    save_intrinsics,
    undistort_image,
    unproject_pixel,
)
from src.calibration.runner import resolve_calibration

__all__ = [
    "CalibrationResult",
    "Intrinsics",
    "calibrate_charuco",
    "calibrate_checkerboard",
    "detect_checkerboard",
    "load_intrinsics",
    "make_intrinsics",
    "project",
    "resolve_calibration",
    "save_intrinsics",
    "undistort_image",
    "unproject_pixel",
]