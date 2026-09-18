"""STEP 4 — Charuco board calibration.

Charuco = chessboard squares with ArUco markers in the white cells. The
markers are detected first (robust to partial occlusion), then chessboard
corner positions are interpolated from the marker poses — giving sub-pixel
corners like a checkerboard but with less scene area required. Calibration
runs through ``cv2.aruco.calibrateCameraCharuco``.

Both the modern (OpenCV >= 4.7) and legacy ArUco APIs are supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.calibration.checkerboard import CalibrationResult
from src.calibration.intrinsics import Intrinsics
from src.common.logging_utils import get_logger

log = get_logger("sp3d.calibration")


def aruco_dictionary():
    """Predefined 4x4, 50-id dictionary (modern then legacy API)."""
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)


def charuco_board(rows: int, cols: int, square_length_m: float,
                  marker_length_m: float):
    """Build a ``CharucoBoard`` across OpenCV versions.

    ``cols`` = squares per row, ``rows`` = squares per column (marker grid).
    """
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard((cols, rows), square_length_m,
                                      marker_length_m, aruco_dictionary())
    return cv2.aruco.CharucoBoard_create(cols, rows, square_length_m,
                                         marker_length_m, aruco_dictionary())


def generate_charuco_view(board, width: int, height: int) -> np.ndarray:
    """Render a full board image (for tests / demo boards)."""
    if hasattr(board, "generateImage"):
        img = np.zeros((height, width), np.uint8)
        board.generateImage((width, height), img, marginSize=0, borderBits=1)
        return img
    return board.draw((width, height))


def _charuco_detector(board, camera_matrix=None, dist_coeffs=None):
    """Build a ``CharucoDetector`` (modern) or legacy detector+interpolate pair."""
    if hasattr(cv2.aruco, "CharucoDetector"):
        params = cv2.aruco.CharucoParameters()
        params.tryRefineMarkers = True
        if camera_matrix is not None and dist_coeffs is not None:
            params.cameraMatrix = camera_matrix
            params.distCoeffs = dist_coeffs
        return cv2.aruco.CharucoDetector(board, params)
    return None


def detect_charuco(image: np.ndarray, board, camera_matrix=None,
                   dist_coeffs=None):
    """Return (charuco_corners, charuco_ids) or (None, None)."""
    detector = _charuco_detector(board, camera_matrix, dist_coeffs)
    if detector is not None:
        res = detector.detectBoard(image)
        corners, ids = res[0], res[1]
        if corners is None or len(corners) == 0:
            return None, None
        return corners, np.asarray(ids, np.int32).reshape(-1, 1)
    params = cv2.aruco.DetectorParameters()
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        image, aruco_dictionary(), parameters=params)
    kwargs = {"cameraMatrix": camera_matrix, "distCoeffs": dist_coeffs} \
        if camera_matrix is not None and dist_coeffs is not None else {}
    corners, ids, _ = cv2.aruco.interpolateCornersCharuco(
        marker_corners, image, board, **kwargs)
    if corners is None or len(corners) == 0:
        return None, None
    return corners, np.asarray(ids, np.int32).reshape(-1, 1)