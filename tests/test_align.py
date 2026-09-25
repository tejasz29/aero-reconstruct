"""STEP 8 — visual<->GPS similarity alignment tests.

Ground truths: random visual clouds (seed-pinned) mapped through a known
s/R/t must be recovered to <1% scale and <0.5 m RMSE; RANSAC must survive
30% outliers; timestamp interpolation must not extrapolate.
"""

from __future__ import annotations

import numpy as np

from src.georef.align import (
    SimilarityTransform,
    apply_similarity,
    estimate_similarity_umeyama,
)


def _known_transform() -> SimilarityTransform:
    angle = np.deg2rad(30.0)
    R = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                  [np.sin(angle), np.cos(angle), 0.0],
                  [0.0, 0.0, 1.0]])
    return SimilarityTransform(scale=2.5, rotation=R,
                               translation=np.array([10.0, -5.0, 3.0]))


def _visual_cloud(n=20, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-10.0, 10.0, size=(n, 3))


def test_umeyama_recovers_known_similarity():
    gt = _known_transform()
    src = _visual_cloud()
    dst = apply_similarity(src, gt)
    est = estimate_similarity_umeyama(src, dst)
    assert abs(est.scale - 2.5) / 2.5 < 0.01
    np.testing.assert_allclose(est.R, gt.R, atol=1e-6)
    np.testing.assert_allclose(est.t, gt.t, atol=1e-6)


def test_umeyama_rejects_degenerate_and_short_inputs():
    import pytest

    from src.georef.align import estimate_similarity_umeyama as ume

    with pytest.raises(ValueError, match=">=3"):
        ume(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="degenerate"):
        ume(np.zeros((5, 3)), np.ones((5, 3)))


def test_ransac_survives_outliers():
    from src.georef.align import compute_residuals_m, ransac_similarity

    gt = _known_transform()
    src = _visual_cloud(n=30, seed=1)
    dst = apply_similarity(src, gt)
    dst[::3] += np.array([50.0, -40.0, 30.0])  # ~33% gross outliers
    est, inliers = ransac_similarity(src, dst, iterations=500,
                                     inlier_threshold_m=0.5,
                                     min_correspondences=6, seed=42)
    assert inliers.sum() >= 18
    assert abs(est.scale - 2.5) / 2.5 < 0.01
    err = compute_residuals_m(src[inliers], dst[inliers], est)
    assert float((err ** 2).mean() ** 0.5) < 0.5


def test_interpolation_no_extrapolation_and_rotation_math():
    import pytest

    from src.georef.align import (
        MetricSample,
        VisualSample,
        align_camera_rotation,
        interpolate_metric_to_visual,
        is_valid_rotation,
    )

    visual = [VisualSample(t, np.array([float(t), 0.0, 0.0]))
              for t in (0.0, 0.5, 1.0, 5.0)]
    metric = [MetricSample(t, np.array([2.0 * t, 0.0, 0.0]))
              for t in (0.0, 1.0)]
    src, dst, kept = interpolate_metric_to_visual(visual, metric)
    assert len(kept) == 3  # t=5.0 outside GPS span is dropped
    np.testing.assert_allclose(dst[:, 0], [0.0, 1.0, 2.0], atol=1e-9)

    gt = _known_transform()
    assert is_valid_rotation(gt.R)
    assert not is_valid_rotation(np.eye(3) * 2.0)
    R_wc = np.eye(3)
    np.testing.assert_allclose(align_camera_rotation(R_wc, gt), gt.R.T, atol=1e-9)
    with pytest.raises(ValueError, match="non-empty"):
        interpolate_metric_to_visual([], metric)


def _write_poses_and_gps(tmp_path, transform: SimilarityTransform, n=12):
    import csv

    src = _visual_cloud(n=n, seed=7)
    dst = apply_similarity(src, transform)
    poses = tmp_path / "camera_poses.csv"
    with open(poses, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_id", "source_index", "timestamp_s", "filename",
                    "kept", "reject_reason", "tx", "ty", "tz",
                    "r00", "r01", "r02", "r10", "r11", "r12",
                    "r20", "r21", "r22", "inliers",
                    "mean_reproj_error_px", "median_parallax_px"])
        for i in range(n):
            w.writerow([i, i, f"{i * 0.5:.6f}", f"frame_{i:06d}.jpg",
                        1, "", f"{src[i, 0]:.6f}", f"{src[i, 1]:.6f}",
                        f"{src[i, 2]:.6f}",
                        1, 0, 0, 0, 1, 0, 0, 0, 1, 50, 0.5, 10.0])
    gps = tmp_path / "gps_metric.csv"
    with open(gps, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp_s", "latitude", "longitude", "altitude_m",
                    "easting_m", "northing_m", "up_m", "crs", "zone"])
        for i in range(n):
            w.writerow([f"{i * 0.5:.6f}", 47.6, -122.3, 100.0,
                        f"{dst[i, 0]:.6f}", f"{dst[i, 1]:.6f}",
                        f"{dst[i, 2]:.6f}", "enu", ""])
    return poses, gps


def test_run_alignment_e2e_recovers_scale(tmp_path):
    import json

    from src.georef.align import run_alignment

    gt = _known_transform()
    poses, gps = _write_poses_and_gps(tmp_path, gt)
    cfg = {"paths": {"trajectory": str(tmp_path),
                     "georef": str(tmp_path / "georef"),
                     "reports": str(tmp_path / "reports")},
           "alignment": {"ransac_iterations": 200,
                         "inlier_threshold_m": 1.0,
                         "min_correspondences": 6,
                         "use_rtk_if_available": True},
           "project": {"seed": 42}}
    result = run_alignment(cfg, poses_csv=poses, gps_metric_csv=gps,
                           output_dir=tmp_path / "georef")
    assert abs(result.transform.scale - 2.5) / 2.5 < 0.01
    assert result.n_inliers == result.n_correspondences == 12
    assert result.rmse_inliers_m is not None and result.rmse_inliers_m < 0.5
    assert (tmp_path / "georef" / "aligned_trajectory.csv").is_file()
    assert (tmp_path / "georef" / "similarity_transform.json").is_file()
    report = json.loads((tmp_path / "reports" / "alignment_report.json")
                        .read_text(encoding="utf-8"))
    assert report["crs"] == "enu" and report["n_inliers"] == 12


def test_align_cli_parser_and_errors(tmp_path):
    import pytest

    from src import cli
    from src.georef.align import run_alignment

    parser = cli.build_parser()
    args = parser.parse_args(["align-trajectory", "--poses-csv", "p.csv",
                              "--gps-metric-csv", "g.csv",
                              "--output-dir", "o",
                              "--iterations", "100",
                              "--threshold", "1.5"])
    assert args.poses_csv == "p.csv" and args.iterations == 100
    assert "align-trajectory" in cli.build_parser().format_help()
    cfg = {"paths": {}, "alignment": {}, "project": {}}
    with pytest.raises((FileNotFoundError, ValueError)):
        run_alignment(cfg, poses_csv=tmp_path / "missing.csv",
                      gps_metric_csv=tmp_path / "missing2.csv")
