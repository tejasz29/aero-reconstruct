"""STEP 10 — runner e2e + CLI (2-frame synthetic, merged + per-frame)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.fusion.io import load_ply, read_cloud_index
from src.fusion.runner import resolve_paths, resolve_policy, run_unprojection


def _make_project(tmp_path: Path, n=2, scale=2.0):
    from PIL import Image

    from src.calibration.intrinsics import save_intrinsics, make_intrinsics
    from src.sfm.runner import POSES_HEADER

    frames = tmp_path / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    # images
    for i in range(n):
        arr = (rng.random((12, 16, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(frames / f"f{i}.jpg")
    # depth maps + index
    with open(depth_dir / "depth_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame_id", "filename", "width",
                                           "height", "depth_path",
                                           "confidence_path", "depth_min",
                                           "depth_max", "depth_mean",
                                           "confidence_mean"])
        w.writeheader()
        for i in range(n):
            d = (1.0 + rng.random((12, 16))).astype(np.float32)
            p = depth_dir / f"depth_f{i}.npy"
            np.save(p, d)
            w.writerow({"frame_id": i, "filename": f"f{i}.jpg", "width": 16,
                        "height": 12, "depth_path": str(p), "confidence_path": "",
                        "depth_min": 1.0, "depth_max": 2.0, "depth_mean": 1.5,
                        "confidence_mean": 1.0})
    # camera
    intr = make_intrinsics(200.0, 200.0, 8.0, 6.0, 16, 12, source="provided")
    cam = tmp_path / "camera.yaml"
    save_intrinsics(intr, cam)
    # poses (identity + x-shifted)
    poses_csv = tmp_path / "camera_poses.csv"
    with open(poses_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(POSES_HEADER)
        for i in range(n):
            R = np.eye(3)
            C = np.array([float(i), 0.0, 0.0])
            w.writerow([i, i, f"{0.5 * i:.2f}", f"f{i}.jpg", 1, ""]
                       + [f"{v:.6f}" for v in C]
                       + [f"{v:.6f}" for v in R.ravel()]
                       + [100, "0.3", "7.0"])
    # transform (scale only)
    transform = tmp_path / "transform.json"
    transform.write_text(json.dumps({"scale": scale,
                                     "rotation": np.eye(3).tolist(),
                                     "translation_m": [0.0, 0.0, 0.0]}),
                         encoding="utf-8")
    return frames, depth_dir / "depth_index.csv", poses_csv, cam, transform


def _cfg(tmp_path):
    from src.common.config_loader import load_config

    cfg = load_config()
    cfg["paths"]["reports"] = str(tmp_path / "reports")
    return cfg


def test_policy_and_paths_from_config(tmp_path):
    cfg = _cfg(tmp_path)
    policy = resolve_policy(cfg)
    assert policy.stride == 2 and policy.save_per_frame is True
    paths = resolve_paths(cfg, output_dir=tmp_path / "cloud")
    assert paths.scene_ply.name == "scene.ply"
    assert paths.report_json.name == "unproject_report.json"


def test_run_unprojection_e2e(tmp_path):
    frames, depth_idx, poses_csv, cam, transform = _make_project(tmp_path)
    cfg = _cfg(tmp_path)
    result = run_unprojection(cfg, depth_index=depth_idx, poses_csv=poses_csv,
                              camera_yaml=cam, transform_json=transform,
                              frames_dir=frames,
                              output_dir=tmp_path / "cloud", stride=2)
    assert result["n_frames"] == 2
    assert result["scale"] == pytest.approx(2.0)
    assert Path(result["scene_ply"]).is_file()
    rows = read_cloud_index(result["index_csv"])
    assert len(rows) == 2 and all(r["ply_path"] for r in rows)
    pts, cols = load_ply(result["scene_ply"])
    assert len(pts) == 2 * 6 * 8  # stride 2 on 12x16
    assert result["stats"]["n_points"] == len(pts)
    report = json.loads(open(result["report_json"]).read())
    assert report["metric_via_gps_scale"] is True
    assert report["absolute_crs"] is False
    assert "NOT yet" in report["note"] or "NOT absolute" in report["note"] or "STEP 16" in report["note"]


def test_run_unprojection_missing_transform_uses_scale_one(tmp_path):
    frames, depth_idx, poses_csv, cam, _ = _make_project(tmp_path)
    cfg = _cfg(tmp_path)
    result = run_unprojection(cfg, depth_index=depth_idx, poses_csv=poses_csv,
                              camera_yaml=cam,
                              transform_json=tmp_path / "no.json",
                              frames_dir=frames,
                              output_dir=tmp_path / "cloud1", stride=1)
    assert result["scale"] == pytest.approx(1.0)


def test_run_unprojection_requires_inputs(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises((FileNotFoundError, ValueError)):
        run_unprojection(cfg, depth_index=tmp_path / "no.csv",
                         poses_csv=tmp_path / "no2.csv",
                         camera_yaml=tmp_path / "no3.yaml",
                         frames_dir=tmp_path, output_dir=tmp_path / "out")


def test_unproject_cli_parser_and_e2e(tmp_path, capsys):
    from src import cli

    parser = cli.build_parser()
    assert "unproject-depth" in parser.format_help()
    args = parser.parse_args(["unproject-depth", "--stride", "2"])
    assert args.stride == 2

    frames, depth_idx, poses_csv, cam, transform = _make_project(tmp_path)
    out = tmp_path / "cloud_cli"
    code = cli.main(["unproject-depth", "--depth-index", str(depth_idx),
                     "--poses-csv", str(poses_csv), "--camera", str(cam),
                     "--transform-json", str(transform),
                     "--frames-dir", str(frames),
                     "--output-dir", str(out), "--stride", "2"])
    assert code == 0
    text = capsys.readouterr().out
    assert "merged" in text and "NOT absolute CRS" in text
