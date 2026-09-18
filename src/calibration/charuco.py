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


def calibrate_charuco(
    images: list[str | Path],
    rows: int,
    cols: int,
    square_length_m: float,
    marker_length_m: float,
    image_size: tuple[int, int] | None = None,
) -> CalibrationResult:
    """Calibrate intrinsics from a list of Charuco board photos."""
    board = charuco_board(rows, cols, square_length_m, marker_length_m)
    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    used: list[Path] = []
    rejected: list[tuple[Path, str]] = []
    resolved_size: tuple[int, int] | None = None

    for img_path in tqdm(images, desc="charuco", unit="img"):
        img_path = Path(img_path)
        image = cv2.imread(str(img_path))
        if image is None:
            rejected.append((img_path, "unreadable"))
            continue
        if resolved_size is None or image_size is not None:
            resolved_size = image_size or (image.shape[1], image.shape[0])
        corners, ids = detect_charuco(image, board)
        if corners is None or len(corners) < 4:
            rejected.append((img_path, "not_enough_charuco_corners"))
            continue
        all_corners.append(corners)
        all_ids.append(ids)
        used.append(img_path)

    if resolved_size is None:
        raise ValueError("no readable images supplied for calibration")
    if len(used) < 3:
        raise ValueError(
            f"charuco needs >= 3 views, only {len(used)} detected "
            f"(rejected={len(rejected)})")

    # Build object points once (charuco chess corners are a fixed grid) and
    # calibrate with plain calibrateCamera. Works on every OpenCV that supports
    # CharucoBoard.getChessboardCorners (>= 4.7).
    #
    # Two id conventions exist: classic API returns the *marker id* per corner
    # (-> lookup in the marker-id array), while the newer API returns the
    # *index* of the chess corner in the board grid (-> direct indexing).
    board_corners = np.asarray(board.getChessboardCorners(), np.float32)
    if hasattr(board, "getChessboardIds"):
        board_ids = np.asarray(board.getChessboardIds()).reshape(-1)
        lookup = {int(bid): board_corners[idx] for idx, bid in enumerate(board_ids)}
        resolve = lambda ids: np.asarray(
            [lookup[int(i)] for i in np.asarray(ids).flatten()], np.float32)
    else:
        resolve = lambda ids: np.take(
            board_corners, np.asarray(ids, np.int32).flatten(), axis=0)
    obj_list: list[np.ndarray] = []
    img_list: list[np.ndarray] = []
    for corners, ids in zip(all_corners, all_ids):
        obj_list.append(resolve(ids).reshape(-1, 3))
        img_list.append(corners.reshape(-1, 2))

    flags = 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, resolved_size, None, None, flags=flags)

    intrinsics = Intrinsics(
        fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
        width=resolved_size[0], height=resolved_size[1],
        distortion=tuple(float(d) for d in np.asarray(dist).ravel()),
        source="charuco", reprojection_error_px=float(rms),
        calibrated_on=__import__("time").strftime("%Y-%m-%d"),
    ).validate()

    # Per-view RMS: reproject each view's charuco object points (already
    # matched above via the board corner lookup) with the fitted pose.
    per_view: list[tuple[Path, float, int]] = []
    for path, obj, corners, rvec, tvec in zip(used, obj_list, all_corners, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        sq_err = np.sum((projected.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1)
        per_view.append((path, float(np.sqrt(float(np.mean(sq_err)))), len(corners)))

    log.info("charuco calibration: rms=%.3f px across %d views",
             float(rms), len(used))
    return CalibrationResult(
        intrinsics=intrinsics, rms_px=float(rms),
        used_images=used, rejected_images=rejected, per_view_rms_px=per_view,
        method="charuco")