"""STEP 10 — unprojection math tests (round-trip, pose, scale)."""

from __future__ import annotations

import numpy as np
import pytest

from src.fusion.unproject import (apply_pose, apply_scale, pixel_grid,
                                  reproject_points, unproject_depth,
                                  valid_mask)


def test_pixel_grid_shape_and_origin():
    us, vs = pixel_grid(4, 3)
    assert us.shape == (3, 4) and vs.shape == (3, 4)
    assert us[0, 0] == 0.0 and vs[0, 0] == 0.0
    assert us[0, 3] == 3.0 and vs[2, 0] == 2.0
    with pytest.raises(ValueError):
        pixel_grid(0, 4)


def test_unproject_reproject_round_trip():
    fx = fy = 500.0
    cx, cy = 16.0, 12.0
    rng = np.random.default_rng(0)
    depth = 1.0 + rng.random((24, 32))
    cam = unproject_depth(depth, fx, fy, cx, cy)
    assert cam.shape == (24, 32, 3)
    assert bool(np.all(np.isfinite(cam)))
    # centre pixel maps to (0, 0, Z)
    assert cam[12, 16, 0] == pytest.approx(0.0, abs=1e-9)
    # round-trip every 4th pixel
    flat = cam[::4, ::4].reshape(-1, 3)
    uv = reproject_points(flat, fx, fy, cx, cy)
    uu, vv = np.meshgrid(np.arange(32)[::4], np.arange(24)[::4])
    assert np.allclose(uv[:, 0], uu.ravel(), atol=1e-6)
    assert np.allclose(uv[:, 1], vv.ravel(), atol=1e-6)


def test_unproject_rejects_bad_focal_and_shape():
    with pytest.raises(ValueError):
        unproject_depth(np.ones((4, 4)), 0.0, 500.0, 2.0, 2.0)
    with pytest.raises(ValueError):
        unproject_depth(np.ones(4), 500.0, 500.0, 2.0, 2.0)


def test_apply_pose_identity_and_translation():
    pts = np.array([[1.0, 0.0, 5.0], [0.0, 2.0, 3.0]])
    out = apply_pose(pts, np.eye(3), np.zeros(3))
    assert np.allclose(out, pts)
    out2 = apply_pose(pts, np.eye(3), np.array([10.0, 0.0, 0.0]))
    assert np.allclose(out2, pts + [10.0, 0.0, 0.0])


def test_apply_pose_known_rotation():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # world [1,0,0] with C=0 -> cam [0,1,0]; invert back to world
    cam = np.array([[0.0, 1.0, 2.0]])
    world = apply_pose(cam, R, np.zeros(3))
    assert np.allclose(world, [[1.0, 0.0, 2.0]], atol=1e-9)


def test_apply_scale_contract():
    pts = np.ones((3, 3))
    assert np.allclose(apply_scale(pts, 2.5), 2.5 * np.ones((3, 3)))
    with pytest.raises(ValueError):
        apply_scale(pts, 0.0)
    with pytest.raises(ValueError):
        apply_scale(pts, float("nan"))


def test_valid_mask_gates():
    d = np.array([[1.0, -1.0], [np.nan, np.inf]], dtype=np.float32)
    mask = valid_mask(d)
    assert mask.tolist() == [[True, False], [False, False]]
    big = np.array([[100.0]])
    assert not valid_mask(big, max_depth=10.0).item()
