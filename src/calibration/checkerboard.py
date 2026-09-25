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


def calibrate_checkerboard(
    images: list[str | Path],
    pattern_size: tuple[int, int],
    square_size_m: float,
    image_size: tuple[int, int] | None = None,
) -> CalibrationResult:
    """Calibrate from a list of checkerboard photos.

    ``pattern_size = (cols, rows)`` = inner corners per row/column (OpenCV
    convention; e.g. (9, 6) for a classic 9x6 board).
    """
    objp = checkerboard_object_points(pattern_size, square_size_m)
    object_pts: list[np.ndarray] = []
    image_pts: list[np.ndarray] = []
    used: list[Path] = []
    rejected: list[tuple[Path, str]] = []
    resolved_size: tuple[int, int] | None = None

    for img_path in tqdm(images, desc="checkerboard", unit="img"):
        img_path = Path(img_path)
        image = cv2.imread(str(img_path))
        if image is None:
            rejected.append((img_path, "unreadable"))
            continue
        if resolved_size is None or image_size is not None:
            resolved_size = image_size or (image.shape[1], image.shape[0])
        corners = detect_checkerboard(image, pattern_size)
        if corners is None:
            rejected.append((img_path, "pattern_not_detected"))
            continue
        object_pts.append(objp)
        image_pts.append(corners)
        used.append(img_path)

    if resolved_size is None:
        raise ValueError("no readable images supplied for calibration")
    if len(used) < 3:
        raise ValueError(
            f"checkerboard needs >= 3 views, only {len(used)} detected "
            f"(rejected={len(rejected)})")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_pts, image_pts, resolved_size, None, None)

    intrinsics = Intrinsics(
        fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
        width=resolved_size[0], height=resolved_size[1],
        distortion=tuple(float(d) for d in np.asarray(dist).ravel()),
        source="checkerboard", reprojection_error_px=float(rms),
        calibrated_on=__import__("time").strftime("%Y-%m-%d"),
    ).validate()

    per_view: list[tuple[Path, float, int]] = []
    for path, obj, img, rvec, tvec in zip(used, object_pts, image_pts, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img) ** 2,
                                           axis=1))))
        per_view.append((path, err, len(img)))

    log.info("checkerboard calibration: rms=%.3f px across %d views",
             float(rms), len(used))
    return CalibrationResult(
        intrinsics=intrinsics, rms_px=float(rms),
        used_images=used, rejected_images=rejected, per_view_rms_px=per_view,
        method="checkerboard", pattern_size=pattern_size)