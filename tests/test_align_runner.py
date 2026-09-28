"""STEP 8 — alignment runner tests (config policy, paths, end-to-end).

The runner is what the CLI and the API call, so these tests drive it the way
a user does: write a synthetic STEP 5 pose CSV and a synthetic STEP 7 metric
CSV, then check the three artefacts and the report. Ground truth is a known
similarity, so a regression in the fit shows up as a wrong scale rather than
as a vague "something was written".
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from src.georef.align import SimilarityTransform, apply_similarity, transform_from_dict
from src.georef.gps import MetricFix
from src.georef.runner import (
    AlignmentPolicy,
    Correspondences,
    build_correspondences,
    fit_alignment,
    resolve_paths,
    resolve_policy,
    run_alignment,
)
from src.sfm.runner import POSES_HEADER

TRUE_SCALE = 0.45
TRUE_R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
TRUE_T = np.array([310.0, -125.0, 40.0])
TRUE = SimilarityTransform(TRUE_SCALE, TRUE_R, TRUE_T)


def flight_point(i):
    return (float(i), 0.7 * np.sin(i), 0.2 * i)


def write_poses(path, n=18, drop=()):
    """A STEP 5 style pose CSV: a helix-ish pass with some rejected frames."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(POSES_HEADER)
        for i in range(n):
            kept = i not in drop
            row = [i, i, f"{0.5 * i:.6f}", f"frame_{i:03d}.jpg", int(kept), ""]
            if kept:
                C = flight_point(i)
                R = np.eye(3)
                R[0, 0] = np.cos(0.1 * i)
                R[0, 1] = -np.sin(0.1 * i)
                R[1, 0] = np.sin(0.1 * i)
                R[1, 1] = np.cos(0.1 * i)
                row += [f"{v:.6f}" for v in C]
                row += [f"{v:.6f}" for v in R.ravel()]
                row += [120, "0.42", "9.0"]
            else:
                row += [""] * (len(POSES_HEADER) - len(row))
            writer.writerow(row)
    return path


def write_gps(path, n=18, spike_at=None, spike_m=12.0, fix_types=None):
    """A STEP 7 style metric CSV, optionally with an RTK status column."""
    extra = ["fix_type"] if fix_types else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp_s", "latitude", "longitude", "altitude_m",
                         "easting_m", "northing_m", "up_m", "crs", "zone"] + extra)
        for i in range(n):
            e, n_, u = apply_similarity(TRUE, np.asarray(flight_point(i)))[0]
            if spike_at is not None and i == spike_at:
                e += spike_m
            row = [f"{0.5 * i:.6f}", 47.6062, -122.3321, 100.0,
                   f"{e:.6f}", f"{n_:.6f}", f"{u:.6f}", "enu", ""]
            if fix_types:
                row.append(fix_types[i])
            writer.writerow(row)
    return path


@pytest.fixture()
def cfg(tmp_path):
    """A config with every alignment path redirected into tmp_path."""
    from pathlib import Path

    from src.common.config_loader import load_config

    conf = load_config()
    conf["paths"]["trajectory"] = str(tmp_path / "trajectory")
    conf["paths"]["georef"] = str(tmp_path / "georef")
    conf["paths"]["reports"] = str(tmp_path / "reports")
    for key in ("trajectory", "georef", "reports"):
        Path(conf["paths"][key]).mkdir(parents=True, exist_ok=True)
    return conf


@pytest.fixture()
def flight(cfg):
    write_poses(cfg["paths"]["trajectory"] + "/camera_poses.csv", drop={5})
    write_gps(cfg["paths"]["georef"] + "/gps_metric.csv", spike_at=9)
    return cfg


# --- policy + paths ----------------------------------------------------------

def test_policy_comes_from_the_config():
    from src.common.config_loader import load_config

    conf = load_config()
    policy = resolve_policy(conf)
    assert policy.mode == "3d"
    assert policy.iterations == 1000
    assert policy.inlier_threshold_m == 2.0
    assert policy.min_correspondences == 6
    assert policy.min_inliers == 4
    assert policy.as_dict()["rtk_inlier_threshold_m"] == 0.5

    overridden = resolve_policy(conf, mode="2D", inlier_threshold_m=0.75)
    assert overridden.mode == "2d"
    assert overridden.inlier_threshold_m == 0.75


def test_gate_tightens_only_for_corrected_fixes():
    policy = AlignmentPolicy(inlier_threshold_m=2.0, rtk_inlier_threshold_m=0.5)
    from src.georef.align import RtkInfo

    assert policy.gate_m(RtkInfo(10, 4, "rtk")) == 0.5
    assert policy.gate_m(RtkInfo(10, 0, "gps")) == 2.0
    assert policy.gate_m(RtkInfo(10, 0, "unknown")) == 2.0
    # A never-looser gate, whatever the config says.
    loose = AlignmentPolicy(inlier_threshold_m=0.4, rtk_inlier_threshold_m=0.5)
    assert loose.gate_m(RtkInfo(10, 4, "rtk")) == 0.4
    # ...and an RTK run can be told to keep the consumer gate.
    off = AlignmentPolicy(inlier_threshold_m=2.0, use_rtk_if_available=False)
    assert off.gate_m(RtkInfo(10, 4, "rtk")) == 2.0


def test_paths_follow_the_config_and_overrides(tmp_path):
    from src.common.config_loader import load_config

    conf = load_config()
    conf["paths"]["georef"] = str(tmp_path / "geo")
    conf["paths"]["reports"] = str(tmp_path / "rep")
    conf["alignment"]["output_name"] = "traj_utm"
    paths = resolve_paths(conf)
    assert paths.poses_csv.name == "camera_poses.csv"
    assert paths.gps_csv.name == "gps_metric.csv"
    assert paths.aligned_csv == tmp_path / "geo" / "traj_utm.csv"
    assert paths.transform_json == tmp_path / "rep" / "traj_utm_transform.json"
    assert paths.report_json == tmp_path / "rep" / "traj_utm_report.json"
    custom = resolve_paths(conf, poses_csv=tmp_path / "p.csv",
                           gps_csv=tmp_path / "g.csv",
                           output_dir=tmp_path / "out")
    assert custom.poses_csv == tmp_path / "p.csv"
    assert custom.aligned_csv.parent == tmp_path / "out"


# --- correspondence + fit in isolation --------------------------------------

def test_build_correspondences_prefers_rtk_fixes(tmp_path):
    from src.georef.gps import read_metric_table
    from src.sfm.visualize import read_poses_csv

    poses_csv = write_poses(tmp_path / "poses.csv", n=10)
    gps_csv = write_gps(tmp_path / "gps.csv", n=10,
                        fix_types=["RTK_FIXED" if i % 3 else "single"
                                   for i in range(10)])
    poses = read_poses_csv(poses_csv)
    table = read_metric_table(gps_csv)
    corr = build_correspondences(poses, table, AlignmentPolicy())
    assert corr.rtk.source == "rtk"
    assert corr.n_fixes == 6 and corr.n_pairs == 6

    # RTK ignored by policy: every fix is used and the tier is reported.
    off = build_correspondences(poses, table,
                                AlignmentPolicy(use_rtk_if_available=False))
    assert off.n_fixes == 10 and off.n_pairs == 10
    assert off.rtk.source == "rtk"


def test_fit_alignment_rejects_below_the_inlier_floor():
    from src.georef.align import RtkInfo, associate_by_timestamp
    from src.sfm.pose import CameraPose

    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # Targets are the truth plus a large common bias: the fit itself is fine,
    # but three points are below the configured inlier floor.
    target = apply_similarity(TRUE, source) + np.array([9.0, 9.0, 9.0])
    poses = [CameraPose(i, i, 0.5 * i, f"f{i}.jpg", True, R_wc=np.eye(3),
                        C=source[i]) for i in range(3)]
    fixes = [MetricFix(0.5 * i, 47.0, -122.0, 100.0, float(target[i][0]),
                       float(target[i][1]), float(target[i][2]))
             for i in range(3)]
    pairs = associate_by_timestamp(poses, fixes, max_gap_s=0.01)
    corr = Correspondences(pairs=pairs, rtk=RtkInfo(3, 0, "gps"), n_fixes=3,
                           n_poses=3)
    fit = fit_alignment(corr, AlignmentPolicy(min_inliers=4,
                                              inlier_threshold_m=0.1,
                                              iterations=50))
    assert fit.success is False
    assert fit.reject_reason == "insufficient_inliers"
    assert fit.transform.scale == 1.0


# --- end to end --------------------------------------------------------------

def test_run_alignment_end_to_end(flight):
    result = run_alignment(flight)
    assert result.success is True
    assert result.reject_reason == ""
    assert result.mode == "3d"
    assert result.crs == "enu"
    assert result.n_correspondences == 17
    assert result.n_inliers == 16
    assert result.inlier_ratio == pytest.approx(16 / 17, rel=1e-3)
    assert result.scale == pytest.approx(TRUE_SCALE, rel=1e-3)
    assert np.allclose(result.transform.R, TRUE_R, atol=1e-3)
    assert result.rmse_m < 0.05
    assert result.iterations >= 1
    assert result.pairing["n_pairs"] == 17

    aligned = result.aligned_csv
    assert aligned.endswith("aligned_trajectory.csv")
    with open(aligned, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 18
    assert rows[9]["inlier"] == "0"
    assert float(rows[9]["residual_m"]) > 10.0
    assert rows[5]["kept"] == "0" and rows[5]["gx"] == ""

    payload = json.loads(open(result.transform_json, encoding="utf-8").read())
    stored = transform_from_dict(payload)
    assert stored.scale == pytest.approx(TRUE_SCALE, rel=1e-3)
    assert payload["crs"] == "enu"
    assert payload["n_inliers"] == 16

    report = json.loads(open(result.report_json, encoding="utf-8").read())
    assert report["success"] is True
    assert report["crs"] == "enu" and report["mode"] == "3d"
    assert report["inliers"] == 16 and report["correspondences"] == 17
    assert report["residual_m"]["rmse"] < 0.05
    assert report["policy"]["inlier_threshold_m"] == 2.0
    assert report["inputs"]["poses_csv"].endswith("camera_poses.csv")
    assert report["outputs"]["aligned_trajectory_csv"] == aligned
    assert report["pairing"]["n_pairs"] == 17
    # The GPS tier is unknown for a log with no status column, and the report
    # must say so rather than implying a centimetre claim.
    assert report["rtk"]["source"] == "unknown"
    assert "unknown" in report["accuracy"]["statement"]


def test_run_alignment_reports_rtk_tier_and_tight_gate(cfg):
    write_poses(cfg["paths"]["trajectory"] + "/camera_poses.csv", n=18)
    write_gps(cfg["paths"]["georef"] + "/gps_metric.csv", n=18,
              fix_types=["RTK_FIXED"] * 12 + ["single"] * 6)
    result = run_alignment(cfg)
    assert result.success
    assert result.rtk.source == "rtk"
    assert result.n_correspondences == 12
    report = json.loads(open(result.report_json, encoding="utf-8").read())
    assert report["rtk"]["n_fixed"] == 12
    assert report["rtk"]["expected_error_m"] == 0.03
    assert "valid trajectory-accuracy statement" in report["accuracy"]["statement"]


def test_run_alignment_planar_mode(cfg):
    from src.sfm.runner import POSES_HEADER as HEADER

    # Flat nadir pass: mode 2d must recover scale and yaw exactly.
    flat = [(float(i), 0.4 * np.sin(i), 0.0) for i in range(14)]
    with open(cfg["paths"]["trajectory"] + "/camera_poses.csv", "w",
              newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for i in range(14):
            C = flat[i]
            writer.writerow([i, i, f"{0.5 * i:.6f}", f"f{i}.jpg", 1, ""]
                            + [f"{v:.6f}" for v in C]
                            + [f"{v:.6f}" for v in np.eye(3).ravel()]
                            + [100, "0.3", "7.0"])
    with open(cfg["paths"]["georef"] + "/gps_metric.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp_s", "latitude", "longitude", "altitude_m",
                         "easting_m", "northing_m", "up_m", "crs", "zone"])
        for i in range(14):
            e, n_, u = apply_similarity(TRUE, np.asarray(flat[i]))[0]
            writer.writerow([f"{0.5 * i:.6f}", 47.6, -122.3, 100.0,
                             f"{e:.6f}", f"{n_:.6f}", f"{u:.6f}", "enu", ""])
    result = run_alignment(cfg, mode="2d")
    assert result.success
    assert result.mode == "2d"
    assert result.scale == pytest.approx(TRUE_SCALE, rel=1e-6)
    assert result.rmse_m < 1e-6


def test_run_alignment_rejects_too_few_correspondences(cfg):
    write_poses(cfg["paths"]["trajectory"] + "/camera_poses.csv", n=4)
    write_gps(cfg["paths"]["georef"] + "/gps_metric.csv", n=4)
    result = run_alignment(cfg)
    assert result.success is False
    assert result.reject_reason == "too_few_correspondences"
    assert result.aligned_csv == ""
    assert result.scale == 1.0
    report = json.loads(open(result.report_json, encoding="utf-8").read())
    assert report["success"] is False
    assert report["reject_reason"] == "too_few_correspondences"
    assert report["outputs"]["aligned_trajectory_csv"] == ""
    payload = json.loads(open(result.transform_json, encoding="utf-8").read())
    assert payload["status"] == "rejected"
    assert payload["scale"] == 1.0


def test_run_alignment_rejects_a_collinear_trajectory(cfg):
    # A straight-line flight constrains no rotation: refusing is the only
    # honest answer, and it must be reported, not raised.
    with open(cfg["paths"]["trajectory"] + "/camera_poses.csv", "w",
              newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(POSES_HEADER)
        for i in range(10):
            C = (float(i), 0.0, 0.0)
            writer.writerow([i, i, f"{0.5 * i:.6f}", f"f{i}.jpg", 1, ""]
                            + [f"{v:.6f}" for v in C]
                            + [f"{v:.6f}" for v in np.eye(3).ravel()]
                            + [100, "0.3", "7.0"])
    write_gps(cfg["paths"]["georef"] + "/gps_metric.csv", n=10)
    result = run_alignment(cfg)
    assert result.success is False
    assert result.reject_reason in ("no_valid_hypothesis", "insufficient_inliers")
    assert result.scale == 1.0


def test_run_alignment_requires_its_inputs(cfg):
    # Nothing has been written yet: the missing STEP 5/7 inputs must surface
    # as a plain file error rather than a confusing geometry failure.
    with pytest.raises(FileNotFoundError):
        run_alignment(cfg)
