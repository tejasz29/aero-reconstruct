"""STEP 10 — colored_cloud tests (color align, stride, invalid drop)."""

from __future__ import annotations

import numpy as np
import pytest

from src.fusion.unproject import cloud_stats, colored_cloud


def _toy(h=12, w=16, seed=0):
    rng = np.random.default_rng(seed)
    depth = 1.0 + rng.random((h, w)).astype(np.float32)
    rgb = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    conf = (0.5 + 0.5 * rng.random((h, w))).astype(np.float32)
    return depth, rgb, conf


def test_colored_cloud_shapes_and_color_align():
    depth, rgb, conf = _toy()
    pts, cols, out_conf = colored_cloud(depth, rgb, 200.0, 200.0, 8.0, 6.0,
                                        np.eye(3), np.zeros(3), scale=1.0,
                                        confidence=conf, stride=1)
    assert pts.shape == (12 * 16, 3) and cols.shape == (12 * 16, 3)
    assert out_conf.shape == (12 * 16,)
    # first pixel color rides along exactly
    assert (cols[0] == rgb[0, 0]).all()
    assert bool(np.all(np.isfinite(pts)))


def test_colored_cloud_drops_invalid_and_applies_scale():
    depth, rgb, _ = _toy()
    depth[0, 0] = 0.0
    depth[1, 1] = np.nan
    pts1, _, _ = colored_cloud(depth, rgb, 200.0, 200.0, 8.0, 6.0,
                               np.eye(3), np.zeros(3), scale=1.0)
    assert len(pts1) == 12 * 16 - 2
    pts2, _, _ = colored_cloud(depth, rgb, 200.0, 200.0, 8.0, 6.0,
                               np.eye(3), np.zeros(3), scale=3.0)
    assert np.allclose(pts2, pts1 * 3.0)


def test_colored_cloud_stride_thins_and_deterministic():
    depth, rgb, _ = _toy()
    pts1, cols1, _ = colored_cloud(depth, rgb, 200.0, 200.0, 8.0, 6.0,
                                  np.eye(3), np.zeros(3), stride=2)
    pts2, cols2, _ = colored_cloud(depth, rgb, 200.0, 200.0, 8.0, 6.0,
                                  np.eye(3), np.zeros(3), stride=2)
    assert len(pts1) == 6 * 8
    assert np.array_equal(pts1, pts2) and np.array_equal(cols1, cols2)


def test_colored_cloud_rejects_mismatch_and_bad_stride():
    depth, rgb, _ = _toy()
    with pytest.raises(ValueError):
        colored_cloud(depth, rgb[:4], 200.0, 200.0, 8.0, 6.0,
                      np.eye(3), np.zeros(3))
    with pytest.raises(ValueError):
        colored_cloud(depth, rgb, 200.0, 200.0, 8.0, 6.0,
                      np.eye(3), np.zeros(3), stride=0)


def test_cloud_stats_valid_flag():
    pts = np.array([[0.0, 0.0, 1.0], [2.0, 3.0, 4.0]])
    s = cloud_stats(pts)
    assert s["n_points"] == 2 and s["valid"] is True
    empty = cloud_stats(np.zeros((0, 3)))
    assert empty["n_points"] == 0 and empty["valid"] is False
