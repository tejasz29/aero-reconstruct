"""STEP 11 — voxel fusion unit tests (numpy baseline, no open3d needed)."""

import numpy as np

from src.fusion.fuse import voxel_downsample, voxel_keys


def test_voxel_keys_quantize():
    pts = np.array([[0.01, 0.0, 0.0], [0.15, 0.0, 0.0]])
    keys = voxel_keys(pts, 0.10)
    assert keys[0, 0] == 0 and keys[1, 0] == 1


def test_voxel_keys_reject_bad_size():
    import pytest

    with pytest.raises(ValueError):
        voxel_keys(np.ones((2, 3)), 0.0)


def test_voxel_collapses_duplicates():
    pts = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cols = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    p, _c, _f = voxel_downsample(pts, cols, voxel_size=0.1)
    assert len(p) == 2  # first two share a voxel


def test_weighted_average_prefers_confident_point():
    pts = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    cols = np.array([[0, 0, 0], [200, 200, 200]], dtype=np.uint8)
    conf = np.array([0.05, 0.95])
    p, c, _f = voxel_downsample(pts, cols, conf, voxel_size=0.5)
    assert len(p) == 1
    assert p[0, 0] > 0.03  # pulled toward the confident point
    assert c[0, 0] > 150


def test_downsample_deterministic():
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 1, (50, 3))
    cols = (rng.uniform(0, 255, (50, 3))).astype(np.uint8)
    a = voxel_downsample(pts, cols, voxel_size=0.2)
    b = voxel_downsample(pts[::-1], cols[::-1], voxel_size=0.2)
    assert np.allclose(np.sort(a[0].ravel()), np.sort(b[0].ravel()))


def test_downsample_rejects_empty_and_mismatch():
    import pytest

    with pytest.raises(ValueError):
        voxel_downsample(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        voxel_downsample(np.ones((2, 3)),
                         np.ones((3, 3), dtype=np.uint8))


def test_normals_reject_empty():
    import pytest

    from src.fusion.fuse import estimate_normals_pca

    with pytest.raises(ValueError):
        estimate_normals_pca(np.zeros((0, 3)))
