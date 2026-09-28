"""STEP 8 — CLI surface for ``align-trajectory``.

The command has to be discoverable, configurable from the command line, and
honest about a rejected alignment: the exit code is the contract a script or
a CI job reads, so a failed alignment must not exit 0.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from src import cli
from src.georef.align import SimilarityTransform, apply_similarity, transform_from_dict
from src.sfm.runner import POSES_HEADER

TRUE_SCALE = 0.45
TRUE = SimilarityTransform(
    TRUE_SCALE,
    np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    np.array([310.0, -125.0, 40.0]))


def flight_point(i):
    return (float(i), 0.7 * np.sin(i), 0.2 * i)


def write_poses(path, n=16):
    """A STEP 5 style pose CSV, as the CLI would read it."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(POSES_HEADER)
        for i in range(n):
            C = flight_point(i)
            writer.writerow([i, i, f"{0.5 * i:.6f}", f"frame_{i:03d}.jpg", 1, ""]
                            + [f"{v:.6f}" for v in C]
                            + [f"{v:.6f}" for v in np.eye(3).ravel()]
                            + [120, "0.42", "9.0"])
    return path


def write_gps(path, n=16, spike_at=None, spike_m=12.0):
    """A STEP 7 style metric CSV, as the CLI would read it."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp_s", "latitude", "longitude", "altitude_m",
                         "easting_m", "northing_m", "up_m", "crs", "zone"])
        for i in range(n):
            e, n_, u = apply_similarity(TRUE, np.asarray(flight_point(i)))[0]
            if spike_at is not None and i == spike_at:
                e += spike_m
            writer.writerow([f"{0.5 * i:.6f}", 47.6062, -122.3321, 100.0,
                             f"{e:.6f}", f"{n_:.6f}", f"{u:.6f}", "enu", ""])
    return path


def test_align_trajectory_is_listed_in_the_help():
    help_text = cli.build_parser().format_help()
    assert "align-trajectory" in help_text


def test_align_trajectory_parser_exposes_expected_flags():
    args = cli.build_parser().parse_args([
        "align-trajectory", "--poses-csv", "poses.csv",
        "--gps-metric-csv", "gps_metric.csv", "--output-dir", "geo",
        "--mode", "2d", "--inlier-threshold", "0.75"])
    assert args.poses_csv == "poses.csv"
    assert args.gps_metric_csv == "gps_metric.csv"
    assert args.output_dir == "geo"
    assert args.mode == "2d"
    assert args.inlier_threshold == 0.75


def test_align_trajectory_rejects_an_unknown_mode():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["align-trajectory", "--mode", "4d"])


@pytest.fixture()
def run_config(tmp_path):
    """A run config YAML with the alignment inputs pointing at tmp_path."""
    import yaml

    from src.common.config_loader import load_config

    conf = load_config()
    conf["paths"]["trajectory"] = str(tmp_path / "trajectory")
    conf["paths"]["georef"] = str(tmp_path / "georef")
    conf["paths"]["reports"] = str(tmp_path / "reports")
    for key in ("trajectory", "georef", "reports"):
        (tmp_path / key).mkdir(parents=True, exist_ok=True)
    write_poses(str(tmp_path / "trajectory" / "camera_poses.csv"), n=16)
    write_gps(str(tmp_path / "georef" / "gps_metric.csv"), n=16, spike_at=4)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(conf), encoding="utf-8")
    return str(path)


def test_align_trajectory_command_reports_the_fit(run_config, capsys):
    assert cli.main(["align-trajectory", "--config", run_config]) == 0
    out = capsys.readouterr().out
    assert "mode    : 3d" in out
    assert "inliers" in out
    assert f"{TRUE_SCALE:.4f} m per SfM unit" in out
    assert "aligned_trajectory.csv" in out
    assert "gps     : unknown tier" in out


def test_align_trajectory_command_writes_the_three_artefacts(run_config, tmp_path):
    assert cli.main(["align-trajectory", "--config", run_config]) == 0
    aligned = tmp_path / "georef" / "aligned_trajectory.csv"
    assert aligned.is_file()
    with open(aligned, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 16
    assert rows[4]["inlier"] == "0"

    transform = tmp_path / "reports" / "aligned_trajectory_transform.json"
    payload = json.loads(transform.read_text(encoding="utf-8"))
    assert transform_from_dict(payload).scale == pytest.approx(TRUE_SCALE,
                                                               rel=1e-3)

    report = json.loads((tmp_path / "reports"
                         / "aligned_trajectory_report.json").read_text(
                             encoding="utf-8"))
    assert report["success"] is True
    assert report["scale_m_per_sfm_unit"] == pytest.approx(TRUE_SCALE, rel=1e-3)


def test_align_trajectory_exits_non_zero_when_rejected(run_config, capsys):
    import yaml

    from src.common.config_loader import load_config

    conf = load_config(run_config)
    conf["alignment"]["min_correspondences"] = 99
    strict = run_config.replace("run.yaml", "strict.yaml")
    with open(strict, "w", encoding="utf-8") as fh:
        yaml.safe_dump(conf, fh)

    assert cli.main(["align-trajectory", "--config", strict]) == 1
    out = capsys.readouterr().out
    assert "REJECTED (too_few_correspondences)" in out
    assert "NOT georeferenced" in out


def test_align_trajectory_mode_override_uses_the_config_inputs(run_config, capsys):
    # Mode 2d on a 3D pass must still produce a report, and say so.
    assert cli.main(["align-trajectory", "--config", run_config,
                     "--mode", "2d"]) in (0, 1)
    out = capsys.readouterr().out
    assert "mode    : 2d" in out or "REJECTED" in out


def test_alignment_is_reproducible_with_a_fixed_seed(run_config, capsys):
    assert cli.main(["align-trajectory", "--config", run_config]) == 0
    first = capsys.readouterr().out
    assert cli.main(["align-trajectory", "--config", run_config]) == 0
    second = capsys.readouterr().out
    assert first.splitlines()[-3:] == second.splitlines()[-3:]


def test_true_transform_is_the_ground_truth_used_by_the_fixture():
    # Guards the fixture itself: if TRUE were wrong, every assertion above
    # would be passing for the wrong reason.
    point = np.asarray(flight_point(3))
    # R is a +90 deg yaw: (x, y, z) -> (-y, x, z), then scale, then offset.
    expected = np.array([-TRUE.scale * point[1] + TRUE.t[0],
                         TRUE.scale * point[0] + TRUE.t[1],
                         TRUE.scale * point[2] + TRUE.t[2]])
    assert np.allclose(apply_similarity(TRUE, point)[0], expected)
