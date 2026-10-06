"""STEP 10 — PLY + index + report round-trip tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.fusion.io import (load_ply, read_cloud_index, save_ply,
                           validate_cloud, write_cloud_index,
                           write_unproject_report)


def test_ply_roundtrip(tmp_path):
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cols = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    p = save_ply(pts, cols, tmp_path / "a.ply")
    back_pts, back_cols = load_ply(p)
    assert np.allclose(back_pts, pts, atol=1e-6)
    assert (back_cols == cols).all()
    validate_cloud(back_pts, back_cols)


def test_ply_rejects_empty_and_mismatch(tmp_path):
    with pytest.raises(ValueError):
        save_ply(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8),
                 tmp_path / "empty.ply")
    with pytest.raises(ValueError):
        save_ply(np.ones((2, 3)), np.ones((3, 3), dtype=np.uint8),
                 tmp_path / "bad.ply")
    with pytest.raises(FileNotFoundError):
        load_ply(tmp_path / "missing.ply")


def test_validate_cloud_contract():
    good_pts = np.ones((4, 3))
    good_cols = np.ones((4, 3), dtype=np.uint8) * 10
    validate_cloud(good_pts, good_cols)
    with pytest.raises(ValueError):
        validate_cloud(np.ones((4, 2)), good_cols)
    with pytest.raises(ValueError):
        validate_cloud(np.full((2, 3), np.nan), good_cols)


def test_cloud_index_roundtrip(tmp_path):
    rows = [{"frame_id": 0, "filename": "a.jpg", "n_points": 10,
             "ply_path": "a.ply", "mean_confidence": 0.9}]
    idx = write_cloud_index(rows, tmp_path / "cloud_index.csv")
    assert read_cloud_index(idx)[0]["filename"] == "a.jpg"


def test_report_written(tmp_path):
    import json

    rep = write_unproject_report({"n_points": 5}, tmp_path / "rep.json")
    assert json.loads(open(rep).read())["n_points"] == 5
