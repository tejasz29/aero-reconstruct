"""STEP 5 — classical SfM: relative pose, PnP, matching, sequential stitching.

Synthetic ground truth is rendered with the pinhole camera: a random 3D
point cloud is projected through known camera poses, so every recovery /
rejection path is asserted against exact values.
"""

import csv
import json
import shutil

import cv2
import numpy as np
import pytest

from src.calibration.intrinsics import make_intrinsics, save_intrinsics
from src.sfm import runner as sfm_runner
from src.sfm.features import extract_features, match_features
from src.sfm.pose import (
    CameraPose,
    chain_pose,
    estimate_absolute_pose,
    estimate_relative_pose,
)

W, H = 640, 480
K = np.array([[700.0, 0.0, 320.0],
              [0.0, 700.0, 240.0],
              [0.0, 0.0, 1.0]])


def _project(points_cam, camera=K):
    pts = np.asarray(points_cam, dtype=np.float64)
    if pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = pts.T
    pix = pts @ camera.T
    return pix[:, :2] / pix[:, 2:3]


def _rot_angle_deg(R, R_gt):
    R = np.asarray(R, dtype=np.float64)
    R_gt = np.asarray(R_gt, dtype=np.float64)
    cos = float(np.clip((np.trace(R_gt.T @ R) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _point_cloud(n=400, x=(-2.2, 2.2), y=(-2.2, 2.2), z=(4.0, 8.0), seed=7):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(x[0], x[1], n),
        rng.uniform(y[0], y[1], n),
        rng.uniform(z[0], z[1], n),
    ])


def _gt_cameras(n=5, seed=3):
    """World == first camera frame; mostly-forward drone pass with jitter.

    Forward (camera-away) baselines are essential for unambiguous recovery:
    a purely lateral baseline makes the front-depth check symmetric, so the
    recovered rig can be a lateral mirror of the truth.
    """
    rng = np.random.default_rng(seed)
    R_wc, C = [], []
    R_cur, C_cur = np.eye(3), np.zeros(3)
    R_wc.append(R_cur.copy())
    C.append(C_cur.copy())
    for _ in range(1, n):
        t = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.05, 0.05), 0.45])
        rvec = np.array([rng.uniform(-0.01, 0.01), rng.uniform(-0.015, 0.015),
                         rng.uniform(-0.03, 0.03)])
        R_rel, _ = cv2.Rodrigues(rvec)
        R_cur = R_rel @ R_cur
        C_cur = R_rel @ C_cur + t
        R_wc.append(R_cur.copy())
        C.append(C_cur.copy())
    return R_wc, C


def _texture(seed, size=400):
    """Seeded high-frequency texture: dense, unique, SIFT-friendly."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (size, size), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 1.0)


def _warp_plane(texture, plane_corners, R_wc, C, img):
    """Perspective-warp one textured plane into the view (homography path)."""
    th, tw = texture.shape[:2]
    X_cam = (R_wc @ (np.asarray(plane_corners, float) - C).T).T
    pix = _project(X_cam).astype(np.float32)
    src = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
    M = cv2.getPerspectiveTransform(src, pix)
    warped = cv2.warpPerspective(texture, M, (W, H), borderValue=255)
    painted = warped != 255
    img[painted] = warped[painted]


GROUND = np.array([[-2.5, -2.2, 4.0], [2.5, -2.2, 4.0],
                   [2.5, -2.2, 9.0], [-2.5, -2.2, 9.0]])
WALL_R = np.array([[2.2, -2.2, 8.0], [2.2, -2.2, 5.0],
                   [2.2, 2.2, 5.0], [2.2, 2.2, 8.0]])
WALL_L = np.array([[-2.2, 2.2, 8.0], [-2.2, 2.2, 5.0],
                   [-2.2, -2.2, 5.0], [-2.2, -2.2, 8.0]])


def _render_view(points_world, R_wc, C, blob_r=5, sigma=1.8):
    """Render a textured 3-plane scene through pose (R_wc, C)."""
    img = np.full((H, W), 255, np.uint8)
    _warp_plane(_texture(101), WALL_L, R_wc, C, img)
    _warp_plane(_texture(102), WALL_R, R_wc, C, img)
    _warp_plane(_texture(103), GROUND, R_wc, C, img)
    return img


def _write_scene(tmp_path, n_cams=5, black_frame=None):
    """Render a full trajectory: images dir, keyframes.csv, camera.yaml."""
    R_wc, C = _gt_cameras(n=n_cams)
    images = tmp_path / "images"
    images.mkdir()
    for i in range(n_cams):
        img = _render_view(_point_cloud(), R_wc[i], C[i])
        if black_frame == i:
            img = np.full_like(img, 0)
        cv2.imwrite(str(images / f"view_{i:02d}.jpg"), img)
    kf = tmp_path / "keyframes.csv"
    with open(kf, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_id", "source_index", "timestamp_s", "filename"])
        for i in range(n_cams):
            writer.writerow([i, i, f"{0.5 * i + 1.0:.6f}", f"view_{i:02d}.jpg"])
    cam = make_intrinsics(fx=700.0, fy=700.0, cx=320.0, cy=240.0,
                          width=W, height=H, source="synthetic")
    cam_path = save_intrinsics(cam, tmp_path / "camera.yaml")
    return R_wc, C, cam_path


def _cfg(tmp_path):
    return {
        "paths": {"reports": str(tmp_path / "reports")},
        "calibration": {"file": str(tmp_path / "camera.yaml")},
        "sfm": {"backend": "colmap", "feature": "sift", "max_features": 8000,
                "matcher_ratio_test": 0.8,
                "ransac_reproj_threshold_px": 4.0,
                "max_reprojection_error_px": 2.0,
                "min_inliers": 20, "min_parallax_px": 3.0},
    }