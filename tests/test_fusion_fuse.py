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
