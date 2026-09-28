"""STEP 8 — input reading, pose/GPS association, RTK policy, accuracy wording.

The alignment is only as trustworthy as the pairs it is fed, so this file
pins the plumbing in front of the fit:

* reading a STEP 7 ``gps_metric.csv`` back (and noticing a fix-status column),
* pairing poses with fixes one-to-one in time, with a gap gate,
* keeping only RTK/PPK fixes when the log has them,
* and the wording of the accuracy statement, which must not overstate what a
  metre-level GPS track can support.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.georef.align import (
    PoseFixPair,
    RtkInfo,
    SimilarityTransform,
    accuracy_summary,
    apply_similarity,
    associate_by_timestamp,
    build_aligned_rows,
    correspondence_arrays,
    metric_xyz,
    pairing_dt_stats,
    parse_fix_type,
    ransac_similarity,
    rtk_fixed_mask,
    rtk_info_from_types,
    select_rtk_fixes,
    write_aligned_csv,
)
from src.georef.gps import MetricFix, read_metric_csv, read_metric_table
from src.sfm.pose import CameraPose

TRUE_SCALE = 0.85
TRUE_R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
TRUE_T = np.array([50.0, -20.0, 5.0])


def make_pose(frame_id, t, C=(0.0, 0.0, 0.0), kept=True):
    return CameraPose(frame_id, frame_id, t, f"frame_{frame_id:03d}.jpg", kept,
                      R_wc=np.eye(3), C=np.array(C, dtype=float),
                      reject_reason="" if kept else "low_parallax")


def make_fix(t, C, crs_offset=0.0):
    e, n, u = apply_similarity(
        SimilarityTransform(TRUE_SCALE, TRUE_R, TRUE_T), np.asarray(C, float))[0]
    return MetricFix(t, 47.0, -122.0, 100.0, float(e + crs_offset), float(n),
                     float(u))


def flight_point(i):
    """A camera centre that is not collinear — a straight line cannot be fitted."""
    return (float(i), 0.6 * np.sin(i), 0.15 * i)


def write_metric(path, fixes, fix_types=(), crs="enu", zone=""):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp_s", "latitude", "longitude", "altitude_m",
                         "easting_m", "northing_m", "up_m", "crs", "zone"]
                        + (["fix_type"] if fix_types else []))
        for i, f in enumerate(fixes):
            row = [f"{f.timestamp_s:.6f}", f.latitude, f.longitude,
                   f.altitude_m, f"{f.easting_m:.6f}", f"{f.northing_m:.6f}",
                   f"{f.up_m:.6f}", crs, zone]
            if fix_types:
                row.append(fix_types[i])
            writer.writerow(row)
    return path


# --- STEP 7 metric CSV read back -------------------------------------------

def test_metric_table_reads_fixes_crs_zone_and_types(tmp_path):
    fixes = [make_fix(0.0, (0.0, 0.0, 0.0)), make_fix(0.5, (1.0, 2.0, 3.0))]
    path = write_metric(tmp_path / "gps_metric.csv", fixes,
                        fix_types=("RTK_FIXED", "single"), crs="utm", zone="N10")
    table = read_metric_table(path)
    assert table.crs == "utm" and table.zone == "N10"
    assert table.fix_types == ["RTK_FIXED", "single"]
    assert np.allclose(table.fixes[1].easting_m, fixes[1].easting_m)
    assert np.allclose(table.fixes[0].up_m, fixes[0].up_m)
    assert len(read_metric_csv(path)) == 2


def test_metric_table_without_a_status_column_reports_unknown(tmp_path):
    fixes = [make_fix(0.0, (0.0, 0.0, 0.0))]
    path = write_metric(tmp_path / "gps_metric.csv", fixes)
    assert read_metric_table(path).fix_types == [""]


def test_metric_table_validates_its_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="run convert-gps first"):
        read_metric_table(tmp_path / "missing.csv")
    path = tmp_path / "bad.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("timestamp_s,latitude,longitude\n0.0,47.0,-122.0\n")
    with pytest.raises(ValueError, match="missing column"):
        read_metric_table(path)
    empty = tmp_path / "empty.csv"
    write_metric(empty, [])
    with pytest.raises(ValueError, match="no metric GPS fixes"):
        read_metric_table(empty)


# --- fix-status normalisation ------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("RTK_FIXED", "rtk_fixed"), ("rtk-fixed", "rtk_fixed"), ("PPK", "ppk"),
    ("3D Fix", "3d_fix"), ("GGA quality 1", "gga_quality_1"),
    (True, "rtk_fixed"), (False, "single"), (4, "rtk_fixed"), (1, "gps"),
    (None, ""), ("", ""), ("   ", ""),
])
def test_parse_fix_type_normalises_the_usual_spellings(raw, expected):
    assert parse_fix_type(raw) == expected


def test_rtk_info_distinguishes_unknown_from_consumer_gps():
    assert rtk_info_from_types(["", "", ""]).source == "unknown"
    assert rtk_info_from_types(["gps", "single"]).source == "gps"
    assert rtk_info_from_types(["RTK_FIXED", "single", ""]).source == "rtk"
    unknown = rtk_info_from_types(["", ""])
    assert unknown.expected_error_m is None
    assert rtk_info_from_types(["gps", "gps"]).expected_error_m > 1.0
    assert rtk_info_from_types(["RTK_FIXED", "gps"]).expected_error_m < 0.1
    info = rtk_info_from_types(["RTK_FIXED", "gps", "RTK_FIXED"])
    assert info.ratio == pytest.approx(2 / 3)
    assert info.available is True
    assert info.as_dict()["n_fixes"] == 3


def test_rtk_fixed_mask_and_selection():
    types = ["RTK_FIXED", "single", "", "PPK"]
    assert rtk_fixed_mask(types).tolist() == [True, False, False, True]
    fixes = list(range(4))
    assert select_rtk_fixes(fixes, types) == [0, 3]
    # No corrected fix at all: keep everything rather than refuse the log.
    assert select_rtk_fixes(fixes, ["single", "single", "", "single"]) == fixes
    assert select_rtk_fixes(fixes, ["", "", "", ""]) == fixes
    with pytest.raises(ValueError, match="length mismatch"):
        select_rtk_fixes(fixes, ["RTK_FIXED"])


# --- association -------------------------------------------------------------

def test_association_pairs_nearest_fixes_one_to_one():
    poses = [make_pose(0, 0.0, (0.0, 0.0, 0.0)),
             make_pose(1, 0.2, (1.0, 0.0, 0.0)),
             make_pose(2, 0.4, (2.0, 0.0, 0.0))]
    fixes = [make_fix(0.0, (0.0, 0.0, 0.0)), make_fix(0.5, (2.0, 0.0, 0.0))]
    pairs = associate_by_timestamp(poses, fixes, max_gap_s=0.25)
    assert [p.pose.frame_id for p in pairs] == [0, 2]
    assert all(isinstance(p, PoseFixPair) for p in pairs)
    assert pairs[0].dt_s == pytest.approx(0.0)
    assert pairs[1].dt_s == pytest.approx(0.1)


def test_association_drops_pairs_outside_the_time_gate():
    poses = [make_pose(0, 0.0, (0.0, 0.0, 0.0)), make_pose(1, 9.0, (1.0, 0, 0))]
    fixes = [make_fix(0.0, (0.0, 0.0, 0.0)), make_fix(9.4, (1.0, 0.0, 0.0))]
    assert len(associate_by_timestamp(poses, fixes, max_gap_s=0.25)) == 1
    assert len(associate_by_timestamp(poses, fixes, max_gap_s=0.5)) == 2
    with pytest.raises(ValueError, match="max_time_gap_s must be >= 0"):
        associate_by_timestamp(poses, fixes, max_gap_s=-1.0)
    assert associate_by_timestamp(poses, [], max_gap_s=0.5) == []


def test_association_skips_rejected_poses_and_missing_centres():
    poses = [make_pose(0, 0.0, (0.0, 0.0, 0.0)),
             make_pose(1, 0.1, (1.0, 0.0, 0.0), kept=False),
             make_pose(2, 0.2, (2.0, 0.0, 0.0)),
             CameraPose(3, 3, 0.3, "f3.jpg", True)]
    fixes = [make_fix(0.0, (0.0, 0.0, 0.0)), make_fix(0.1, (1.0, 0.0, 0.0)),
             make_fix(0.2, (2.0, 0.0, 0.0)), make_fix(0.3, (3.0, 0.0, 0.0))]
    pairs = associate_by_timestamp(poses, fixes, max_gap_s=0.01)
    assert [p.pose.frame_id for p in pairs] == [0, 2]


def test_association_returns_pose_time_order():
    poses = [make_pose(0, 0.4, (2.0, 0.0, 0.0)), make_pose(1, 0.0, (0.0, 0.0, 0.0))]
    fixes = [make_fix(0.0, (0.0, 0.0, 0.0)), make_fix(0.4, (2.0, 0.0, 0.0))]
    pairs = associate_by_timestamp(poses, fixes, max_gap_s=0.01)
    assert [p.pose.timestamp_s for p in pairs] == [0.0, 0.4]


def test_correspondence_arrays_and_pairing_stats():
    poses = [make_pose(0, 0.0, (0.0, 0.0, 0.0)), make_pose(1, 0.4, (1.0, 2.0, 0.5))]
    fixes = [make_fix(0.05, (0.0, 0.0, 0.0)), make_fix(0.30, (1.0, 2.0, 0.5))]
    pairs = associate_by_timestamp(poses, fixes, max_gap_s=0.5)
    source, target = correspondence_arrays(pairs)
    assert source.shape == (2, 3) and target.shape == (2, 3)
    assert np.allclose(source[1], [1.0, 2.0, 0.5])
    assert np.allclose(metric_xyz(fixes[0]), [fixes[0].easting_m,
                                              fixes[0].northing_m,
                                              fixes[0].up_m])
    stats = pairing_dt_stats(pairs)
    assert stats["n_pairs"] == 2
    assert stats["max_abs_dt_s"] == pytest.approx(0.10, abs=1e-6)
    assert stats["median_abs_dt_s"] == pytest.approx(0.075, abs=1e-6)
    assert pairing_dt_stats([]) == {"n_pairs": 0, "max_abs_dt_s": 0.0,
                                    "median_abs_dt_s": 0.0}
    empty = correspondence_arrays([])
    assert empty[0].shape == (0, 3) and empty[1].shape == (0, 3)


def test_correspondence_arrays_refuse_a_pose_without_a_centre():
    pair = PoseFixPair(CameraPose(0, 0, 0.0, "f.jpg", True),
                       make_fix(0.0, (0.0, 0.0, 0.0)), 0.0)
    with pytest.raises(ValueError, match="has no camera centre"):
        correspondence_arrays([pair])


# --- aligned rows ------------------------------------------------------------

def test_aligned_rows_cover_every_pose_and_name_the_outliers():
    path = [flight_point(i) for i in range(6)]
    poses = [make_pose(i, 0.5 * i, p) for i, p in enumerate(path)]
    poses.append(make_pose(6, 3.0, kept=False))
    fixes = [make_fix(0.5 * i, p) for i, p in enumerate(path)]
    fixes[2] = MetricFix(1.0, 47.0, -122.0, 100.0,
                         fixes[2].easting_m + 30.0, fixes[2].northing_m,
                         fixes[2].up_m)
    pairs = associate_by_timestamp(poses, fixes, max_gap_s=0.01)
    source, target = correspondence_arrays(pairs)
    fit = ransac_similarity(source, target, inlier_threshold_m=1.0,
                            iterations=200, seed=1)
    rows = build_aligned_rows(poses, pairs, fit, fit.transform, "enu")

    assert len(rows) == len(poses)
    assert [r["frame_id"] for r in rows] == [p.frame_id for p in poses]
    assert rows[2]["inlier"] == 0 and float(rows[2]["residual_m"]) > 20.0
    assert rows[0]["inlier"] == 1 and float(rows[0]["residual_m"]) < 0.01
    # The rejected keyframe keeps a row with the fields that are unknown empty.
    assert rows[-1]["kept"] == 0 and rows[-1]["gx"] == ""
    assert rows[0]["crs"] == "enu"
    assert abs(float(rows[0]["gx"]) - float(rows[0]["easting_m"])) < 1e-6
    assert rows[0]["r00"] != ""


def test_aligned_csv_round_trip(tmp_path):
    poses = [make_pose(i, 0.5 * i, flight_point(i)) for i in range(5)]
    fixes = [make_fix(0.5 * i, flight_point(i)) for i in range(5)]
    pairs = associate_by_timestamp(poses, fixes, max_gap_s=0.01)
    source, target = correspondence_arrays(pairs)
    fit = ransac_similarity(source, target, inlier_threshold_m=0.5,
                            iterations=100, seed=2)
    rows = build_aligned_rows(poses, pairs, fit, fit.transform, "enu")
    path = write_aligned_csv(rows, tmp_path / "aligned_trajectory.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        parsed = list(csv.DictReader(fh))
    assert len(parsed) == 5
    assert parsed[0]["inlier"] == "1"
    assert abs(float(parsed[0]["gx"]) - float(parsed[0]["easting_m"])) < 1e-6
    assert float(parsed[0]["dt_s"]) == pytest.approx(0.0, abs=1e-4)


# --- accuracy wording --------------------------------------------------------

def test_accuracy_summary_never_overstates_the_measurement():
    from src.georef.align import AlignmentFit

    fit = AlignmentFit(transform=SimilarityTransform(0.5, TRUE_R, TRUE_T),
                       n_correspondences=10, inlier_mask=np.ones(10, dtype=bool),
                       n_inliers=9, rmse_m=0.4, median_residual_m=0.3,
                       max_residual_m=1.1)
    rtk = accuracy_summary(fit, RtkInfo(10, 8, "rtk"))
    assert "valid trajectory-accuracy statement" in rtk["statement"]
    assert rtk["alignment_rmse_m"] == 0.4
    assert rtk["scale_m_per_sfm_unit"] == pytest.approx(0.5)
    assert rtk["gps_tier"] == "rtk"

    consumer = accuracy_summary(fit, RtkInfo(10, 0, "gps"))
    assert "not absolute accuracy" in consumer["statement"]
    assert consumer["gps_expected_error_m"] > 1.0

    unknown = accuracy_summary(fit, RtkInfo(10, 0, "unknown"))
    assert "unknown" in unknown["statement"]
    assert unknown["gps_expected_error_m"] is None
