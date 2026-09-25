"""STEP 6 — trajectory visualisation tests.

Plotting runs headless (matplotlib Agg) against the STEP 5 CSV schema;
assertions cover CSV round-tripping, camera-axis conventions, projection
math and PNG artefacts — never pixel content.
"""

import numpy as np
import pytest

from src.sfm import runner as sfm_runner
from src.sfm.pose import CameraPose
from src.sfm.visualize import (
    _topdown_coords,
    camera_axes,
    plot_trajectory,
    read_poses_csv,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _poses(n_kept=5, n_rejected=0, dz=0.5, seed=3):
    rng = np.random.default_rng(seed)
    poses = []
    C = np.zeros(3)
    for i in range(n_kept):
        R = np.eye(3)
        pose = CameraPose(i, i, float(i) * 0.5, f"v{i}.jpg", True, R_wc=R,
                          C=C.copy(), inliers=120 - i,
                          mean_reproj_error_px=0.4, median_parallax_px=12.0)
        poses.append(pose)
        C = C + np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), dz])
    for j in range(n_rejected):
        i = n_kept + j
        poses.append(CameraPose(i, i, float(i) * 0.5, f"v{i}.jpg", False,
                                reject_reason="low_parallax"))
    return poses


# --- pose CSV round-trip ---

def test_poses_csv_roundtrip(tmp_path):
    poses = _poses(n_kept=3, n_rejected=1)
    path = tmp_path / "poses.csv"
    sfm_runner._write_poses_csv(path, poses)
    out = read_poses_csv(path)
    assert [p.kept for p in out] == [True, True, True, False]
    assert np.allclose(out[1].C, poses[1].C)
    assert np.allclose(out[1].R_wc, np.eye(3))
    assert out[0].inliers == 120
    assert out[2].timestamp_s == 1.0
    assert out[3].reject_reason == "low_parallax"
    assert out[3].C is None and out[3].R_wc is None


# --- geometry helpers ---

def test_camera_axes_identity():
    right, up, forward = camera_axes(np.eye(3))
    assert np.allclose(right, [1.0, 0.0, 0.0])
    assert np.allclose(up, [0.0, -1.0, 0.0])
    assert np.allclose(forward, [0.0, 0.0, 1.0])


def test_topdown_coords_projections():
    points = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    xs, ys = _topdown_coords(points, "xy")
    assert np.allclose(xs, [1.0, 4.0]) and np.allclose(ys, [2.0, 5.0])
    xs, ys = _topdown_coords(points, "xz")
    assert np.allclose(xs, [1.0, 4.0]) and np.allclose(ys, [3.0, 6.0])
    xs, ys = _topdown_coords(points, "yz")
    assert np.allclose(xs, [2.0, 5.0]) and np.allclose(ys, [3.0, 6.0])


# --- plotting ---

def test_plot_trajectory_writes_pngs(tmp_path):
    plots = plot_trajectory(_poses(n_rejected=1), tmp_path)
    assert plots.three_d_path.is_file() and plots.topdown_path.is_file()
    for path in plots.paths:
        assert path.stat().st_size > 0
        with open(path, "rb") as fh:
            assert fh.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE


def test_plot_topdown_projection_override(tmp_path):
    plots = plot_trajectory(_poses(), tmp_path, topdown_projection="yz")
    assert plots.topdown_path.is_file()


def test_plot_rejects_unknown_projection(tmp_path):
    with pytest.raises(ValueError):
        plot_trajectory(_poses(), tmp_path, topdown_projection="zz")


def test_plot_with_single_kept_pose(tmp_path):
    plots = plot_trajectory(_poses(n_kept=1), tmp_path)
    assert plots.three_d_path.stat().st_size > 0


def test_plot_without_kept_poses_raises(tmp_path):
    with pytest.raises(ValueError, match="no accepted poses"):
        plot_trajectory(_poses(n_kept=0, n_rejected=2), tmp_path)