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