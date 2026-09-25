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
    RelativePose,
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


def _paint_blobs(img, points_world, R_wc, C, blob_r=6):
    """Prick high-frequency foreground blobs into the view (deep parallax)."""
    X_cam = (np.asarray(R_wc, float) @ (np.asarray(points_world, float) - C).T).T
    pix = _project(X_cam)
    front = X_cam[:, 2] > 0.5
    shades = np.linspace(30, 200, 256).astype(np.uint8)
    for i in np.nonzero(front)[0]:
        z = X_cam[i, 2]
        rr = max(1, int(round(blob_r * 5.0 / z)))
        cv2.circle(img, (int(round(pix[i, 0])), int(round(pix[i, 1]))),
                   rr, int(shades[int((X_cam[i, 0] + X_cam[i, 1]) % 256)]), -1)


def _render_view(points_world, R_wc, C, blob_r=5, sigma=1.8):
    """Render a textured 3-plane scene through pose (R_wc, C)."""
    img = np.full((H, W), 255, np.uint8)
    _warp_plane(_texture(101), WALL_L, R_wc, C, img)
    _warp_plane(_texture(102), WALL_R, R_wc, C, img)
    _warp_plane(_texture(103), GROUND, R_wc, C, img)
    _paint_blobs(img, points_world, R_wc, C, blob_r=blob_r)
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


# --- two-view geometry ---

def test_relative_pose_recovers_ground_truth():
    rng = np.random.default_rng(11)
    X1 = np.column_stack([rng.uniform(-2, 2, 150), rng.uniform(-2, 2, 150),
                          rng.uniform(4, 8, 150)])
    t_gt = np.array([0.4, 0.1, -0.05])
    rvec = np.array([0.05, -0.08, 0.12])
    R_gt, _ = cv2.Rodrigues(rvec)
    X2 = R_gt @ X1.T + t_gt.reshape(3, 1)
    p1 = _project(X1) + rng.normal(0, 0.5, (150, 2))
    p2 = _project(X2) + rng.normal(0, 0.5, (150, 2))
    rel = estimate_relative_pose(p1, p2, K)
    assert rel.valid, rel.reject_reason
    assert rel.inlier_count >= 100
    assert _rot_angle_deg(rel.R, R_gt) < 1.5
    assert float(rel.t @ (t_gt / np.linalg.norm(t_gt))) > 0.98


def test_relative_pose_rejects_no_parallax():
    rng = np.random.default_rng(5)
    X = np.column_stack([rng.uniform(-2, 2, 120), rng.uniform(-2, 2, 120),
                         rng.uniform(4, 8, 120)])
    p1 = _project(X)
    rel = estimate_relative_pose(p1, p1.copy(), K, min_parallax_px=8.0)
    assert not rel.valid
    assert rel.reject_reason == "degenerate"


def test_translation_prior_stabilizes_recovery():
    """A prior sign near the left (z-) end of Z keeps the recovery on-spec."""
    rng = np.random.default_rng(11)
    X1 = np.column_stack([rng.uniform(-2, 2, 150), rng.uniform(-2, 2, 150),
                          rng.uniform(4, 8, 150)])
    t_gt = np.array([0.4, 0.1, -0.05])
    R_gt, _ = cv2.Rodrigues(np.array([0.05, -0.08, 0.12]))
    X2 = R_gt @ X1.T + t_gt.reshape(3, 1)
    p1 = _project(X1) + rng.normal(0, 0.5, (150, 2))
    p2 = _project(X2) + rng.normal(0, 0.5, (150, 2))
    rel_prior = estimate_relative_pose(p1, p2, K, transl_prior=np.array([0.0, 0.0, -1.0]))
    assert rel_prior.valid
    assert float(rel_prior.t @ (t_gt / np.linalg.norm(t_gt))) > 0.98


def test_pnp_recovers_world_pose():
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.uniform(-2, 2, 60), rng.uniform(-2, 2, 60),
                         rng.uniform(4, 9, 60)])
    R_gt = np.array([[0.9997, -0.0112, 0.0240],
                     [0.0112, 1.0000, 0.0088],
                     [-0.0240, -0.0088, 0.9997]])
    C_gt = np.array([0.3, -0.2, -0.4])
    p = _project(R_gt @ (X - C_gt).T) + rng.normal(0, 0.3, (60, 2))
    out = estimate_absolute_pose(X, p, K)
    assert out is not None
    R_cw, C, n_inl = out
    assert n_inl >= 55
    assert _rot_angle_deg(R_cw, R_gt) < 0.5
    assert float(np.linalg.norm(C - C_gt)) < 0.05


def test_pnp_needs_minimum_points():
    X = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])
    p = np.array([[320.0, 240.0], [420.0, 240.0], [320.0, 330.0]])
    assert estimate_absolute_pose(X, p, K) is None


# --- chaining ---

def test_chain_matches_reprojection_truth():
    """Chain must invert the rel map: X_cam2 = R_rel X_cam1 + t_rel."""
    rng = np.random.default_rng(1)
    X_w = np.column_stack([rng.uniform(-2, 2, 90), rng.uniform(-2, 2, 90),
                           rng.uniform(4, 9, 90)])
    R0, C0 = np.eye(3), np.zeros(3)
    t_rel = np.array([0.4, 0.1, -0.05])
    R_rel, _ = cv2.Rodrigues(np.array([0.05, -0.08, 0.12]))
    p1 = _project(R0 @ (X_w - C0).T)
    p2 = _project(R_rel @ (R0 @ (X_w - C0).T) + t_rel.reshape(3, 1))
    rel = estimate_relative_pose(p1, p2, K)
    assert rel.valid, rel.reject_reason
    R1, C1 = chain_pose(R0, C0, rel)
    assert _rot_angle_deg(R1, rel.R) < 1e-6
    assert float(np.linalg.norm(C1 - (-rel.R.T @ rel.t))) < 1e-6
    X2_est = rel.R @ (R0 @ (X_w - C0).T) + rel.t.reshape(3, 1)
    X2_chain = R1 @ (X_w - C1).T
    assert float(np.max(np.abs(X2_chain - X2_est))) < 1e-6


def test_chain_pose_unit():
    R0, C0 = np.eye(3), np.zeros(3)
    rel = RelativePose(R=np.eye(3), t=np.zeros(3))
    R1, C1 = chain_pose(R0, C0, rel)
    assert np.allclose(R1, R0)
    assert np.allclose(C1, C0)


# --- features & matching ---

def test_extract_features_sift_and_orb():
    R_wc, C = _gt_cameras(n=1)
    img = _render_view(_point_cloud(), R_wc[0], C[0])
    kp, des = extract_features(img, "sift", max_features=8000)
    assert des is not None and len(kp) > 500
    assert des.shape[0] == len(kp) and des.shape[1] == 128
    kp2, des2 = extract_features(img, "orb", max_features=2000)
    assert des2 is not None and 500 < len(kp2) <= 2000
    assert des2.shape[1] == 32 and des2.dtype == np.uint8


def test_featureless_frame_returns_none():
    kp, des = extract_features(np.zeros((H, W), np.uint8), "sift")
    assert des is None and not kp


def test_match_produces_cross_checked_correspondences():
    R_wc, C = _gt_cameras(n=2)
    X = _point_cloud()
    img0 = _render_view(X, R_wc[0], C[0])
    img1 = _render_view(X, R_wc[1], C[1])
    kp0, des0 = extract_features(img0, "sift")
    kp1, des1 = extract_features(img1, "sift")
    matches = match_features(des0, des1)
    assert len(matches) >= 60
    idx0, idx1 = matches[:, 0], matches[:, 1]
    assert len(set(idx0.tolist())) == len(idx0)
    assert len(set(idx1.tolist())) == len(idx1)
    p0 = np.asarray([kp0[i].pt for i in idx0])
    p1 = np.asarray([kp1[i].pt for i in idx1])
    assert float(np.median(np.linalg.norm(p1 - p0, axis=1))) > 0.0


# --- end-to-end runner ---

def _pose_errs_scale_aligned(cc_kept, cg):
    """Residuals after best-fit global scale (monocular ambiguity).

    The relative translations are unit-norm, so the chained trajectory only
    matches ground truth up to one unknown global scale; aligning that scale
    is the correct way to grade it (handedness/scale land in STEP 8).
    """
    cc = np.asarray(cc_kept, dtype=np.float64)
    cg = np.asarray(cg, dtype=np.float64)
    lam = float(cg.ravel() @ cc.ravel() / (cc.ravel() @ cc.ravel()))
    return float(np.linalg.norm(lam * cc - cg, axis=1).mean()), lam

def test_runner_reconstructs_forward_trajectory(tmp_path):
    R_gt, C_gt, cam_path = _write_scene(tmp_path, n_cams=5)
    res = sfm_runner.run_reconstruction(
        _cfg(tmp_path), frames_dir=tmp_path / "images",
        keyframes_file=tmp_path / "keyframes.csv", intrinsics=cam_path)
    assert res.backend in ("colmap", "opencv")
    assert len(res.poses) == 5 and len(res.kept) >= 4
    kept = res.kept
    assert len(kept) >= 4
    kept = kept[1:]
    for k in range(len(kept)):
        gp = min(range(len(C_gt)), key=lambda i: float(np.linalg.norm(
            np.asarray(C_gt[i]) - np.asarray(kept[k].C, dtype=np.float64))))
        assert _rot_angle_deg(kept[k].R_wc, R_gt[gp]) < 10.0, f"rot pose {k}"
        dot = float(np.asarray(kept[k].C, dtype=np.float64) @ (
            C_gt[gp] / (np.linalg.norm(C_gt[gp]) + 1e-12)))
        assert min(dot, 1.0) > 0.9, f"dir pose {k}"
    mean_err, _lam = _pose_errs_scale_aligned(
        [p.C for p in kept],
        [np.asarray(C_gt[g]) for g in range(len(C_gt))][1:])
    assert mean_err < 0.15, f"scaled position mean {mean_err:.3f}"
    assert res.mean_reproj_error_px is not None
    assert res.mean_reproj_error_px < 1.5
    assert res.poses_path.is_file() and res.report_path.is_file()


def test_runner_rejects_featureless_and_resumes(tmp_path):
    R_gt, C_gt, cam_path = _write_scene(tmp_path, n_cams=3, black_frame=1)
    res = sfm_runner.run_reconstruction(
        _cfg(tmp_path), frames_dir=tmp_path / "images",
        keyframes_file=tmp_path / "keyframes.csv", intrinsics=cam_path)
    rej = res.rejected
    assert len(rej) == 1 and rej[0].reject_reason == "no_features"
    assert rej[0].frame_id == 1
    kept = res.kept
    assert [p.frame_id for p in kept] == [0, 2]
    gp2 = min(range(len(C_gt)), key=lambda i: float(np.linalg.norm(
        np.asarray(C_gt[i]) - np.asarray(kept[-1].C, dtype=np.float64))))
    assert _rot_angle_deg(kept[-1].R_wc, R_gt[gp2]) < 10.0
    mean_err, _lam = _pose_errs_scale_aligned(
        [np.asarray(p.C) for p in kept],
        [np.asarray(C_gt[i]) for i in (0, gp2)])
    assert mean_err < 0.20, f"scaled position mean {mean_err:.3f}"
    assert res.poses_path.is_file()


# --- outputs schema ---

def test_poses_csv_schema(tmp_path):
    R_gt, C_gt, cam_path = _write_scene(tmp_path, n_cams=2)
    res = sfm_runner.run_reconstruction(
        _cfg(tmp_path), frames_dir=tmp_path / "images",
        keyframes_file=tmp_path / "keyframes.csv", intrinsics=cam_path)
    with open(res.poses_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == sfm_runner.POSES_HEADER
    assert len(rows) == 2
    assert rows[0]["frame_id"] == "0" and rows[0]["kept"] == "1"
    assert rows[0]["tx"] == "0.000000" and rows[0]["r00"] == "1.000000"
    assert rows[1]["kept"] == "1"
    assert rows[1]["tx"] != ""
    float(rows[1]["r22"])  # parses as float without raising


def test_trajectory_report_json(tmp_path):
    R_gt, C_gt, cam_path = _write_scene(tmp_path, n_cams=2)
    res = sfm_runner.run_reconstruction(
        _cfg(tmp_path), frames_dir=tmp_path / "images",
        keyframes_file=tmp_path / "keyframes.csv", intrinsics=cam_path)
    report = json.loads(res.report_path.read_text(encoding="utf-8"))
    assert report["keyframes"] == 2 and report["accepted"] == 2
    assert report["backend"] in ("colmap", "opencv")
    assert report["within_threshold"] is True
    assert report["scale_units"].startswith("relative")


def test_backend_falls_back_when_colmap_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sfm_runner.shutil, "which", lambda _: None)
    R_gt, C_gt, cam_path = _write_scene(tmp_path, n_cams=2)
    res = sfm_runner.run_reconstruction(
        _cfg(tmp_path), frames_dir=tmp_path / "images",
        keyframes_file=tmp_path / "keyframes.csv", intrinsics=cam_path)
    assert res.backend == "opencv"
    assert res.mean_reproj_error_px is not None