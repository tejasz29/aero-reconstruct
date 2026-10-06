"""STEP 9 — io + runner e2e (index CSV, report, tiny-image determinism)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.depth.io import (load_depth_npy, read_depth_index, save_confidence_npy,
                          save_depth_npy, save_preview_png, write_depth_index,
                          write_depth_report)
from src.depth.runner import (resolve_paths, resolve_policy,
                              run_depth_prediction)


def _write_frames(frames_dir: Path, n=3):
    from PIL import Image

    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = (rng.random((24, 32, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(frames_dir / f"frame_{i:03d}.jpg")
    kf = frames_dir / "keyframes.csv"
    with open(kf, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_id", "filename", "timestamp_s"])
        for i in range(n):
            w.writerow([i, f"frame_{i:03d}.jpg", f"{0.5 * i:.2f}"])
    return kf


def test_depth_npy_roundtrip(tmp_path):
    d = np.linspace(0.2, 1.8, 24, dtype=np.float32).reshape(4, 6)
    p = save_depth_npy(d, tmp_path / "depth.npy")
    assert np.allclose(load_depth_npy(p), d)
    with pytest.raises(ValueError):
        load_depth_npy(tmp_path / "missing.npy")


def test_confidence_save_validates(tmp_path):
    c = np.ones((4, 4), dtype=np.float32) * 0.5
    save_confidence_npy(c, tmp_path / "c.npy")
    with pytest.raises(ValueError):
        save_confidence_npy(np.ones((2, 2)) * 5.0, tmp_path / "bad.npy")


def test_preview_png_written(tmp_path):
    d = np.arange(16, dtype=np.float32).reshape(4, 4)
    p = save_preview_png(d, tmp_path / "prev.png")
    assert p.is_file() and p.stat().st_size > 0


def test_index_and_report_roundtrip(tmp_path):
    rows = [{"frame_id": 0, "filename": "a.jpg", "width": 32, "height": 24,
             "depth_path": "d.npy", "confidence_path": "c.npy",
             "depth_min": 0.1, "depth_max": 1.0, "depth_mean": 0.5,
             "confidence_mean": 0.9}]
    idx = write_depth_index(rows, tmp_path / "depth_index.csv")
    assert read_depth_index(idx)[0]["filename"] == "a.jpg"
    rep = write_depth_report({"n": 1}, tmp_path / "depth_report.json")
    assert json.loads(open(rep).read())["n"] == 1


def test_policy_and_paths_from_config(tmp_path):
    from src.common.config_loader import load_config

    cfg = load_config()
    policy = resolve_policy(cfg)
    assert policy.input_size == 518 and policy.store_confidence is True
    paths = resolve_paths(cfg, frames_dir=tmp_path / "frames",
                          output_dir=tmp_path / "depth")
    assert paths.index_csv.name == "depth_index.csv"
    assert paths.report_json.name == "depth_report.json"


def test_run_depth_prediction_e2e(tmp_path):
    from src.common.config_loader import load_config

    frames = tmp_path / "frames"
    _write_frames(frames)
    cfg = load_config()
    cfg["paths"]["reports"] = str(tmp_path / "reports")
    out = tmp_path / "depth"
    result = run_depth_prediction(cfg, frames_dir=frames, output_dir=out,
                                  backend="dummy", device="cpu")
    assert result["n_keyframes"] == 3
    assert Path(result["index_csv"]).is_file()
    rows = read_depth_index(result["index_csv"])
    assert len(rows) == 3
    for r in rows:
        depth = load_depth_npy(r["depth_path"])
        assert depth.shape == (24, 32)
        assert bool(np.all(np.isfinite(depth)))
    report = json.loads(open(result["report_json"]).read())
    assert report["relative"] is True and "metric" in report["note"].lower()


def test_run_depth_determinism_on_tiny_input(tmp_path):
    from src.common.config_loader import load_config

    from PIL import Image

    frames = tmp_path / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    tiny = (np.random.default_rng(5).random((8, 8, 3)) * 255).astype(np.uint8)
    Image.fromarray(tiny).save(frames / "tiny.jpg")
    with open(frames / "keyframes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_id", "filename"])
        w.writerow([0, "tiny.jpg"])
    cfg = load_config()
    cfg["paths"]["reports"] = str(tmp_path / "reports")
    r1 = run_depth_prediction(cfg, frames_dir=frames,
                              output_dir=tmp_path / "d1",
                              backend="dummy", device="cpu")
    r2 = run_depth_prediction(cfg, frames_dir=frames,
                              output_dir=tmp_path / "d2",
                              backend="dummy", device="cpu")
    a = load_depth_npy(read_depth_index(r1["index_csv"])[0]["depth_path"])
    b = load_depth_npy(read_depth_index(r2["index_csv"])[0]["depth_path"])
    assert np.array_equal(a, b)


def test_run_depth_requires_keyframes(tmp_path):
    from src.common.config_loader import load_config

    cfg = load_config()
    cfg["paths"]["reports"] = str(tmp_path / "reports")
    with pytest.raises(FileNotFoundError):
        run_depth_prediction(cfg, frames_dir=tmp_path / "empty",
                             keyframes_file=tmp_path / "no.csv",
                             output_dir=tmp_path / "out")
