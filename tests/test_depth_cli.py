"""STEP 9 — CLI surface for predict-depth."""

from __future__ import annotations

from src import cli


def test_predict_depth_parser_exposes_expected_flags():
    parser = cli.build_parser()
    args = parser.parse_args(["predict-depth", "--frames-dir", "frames",
                              "--keyframes-file", "kf.csv",
                              "--output-dir", "depth",
                              "--backend", "dummy", "--device", "cpu"])
    assert args.frames_dir == "frames"
    assert args.keyframes_file == "kf.csv"
    assert args.output_dir == "depth"
    assert args.backend == "dummy"
    assert args.device == "cpu"


def test_parser_lists_predict_depth():
    assert "predict-depth" in cli.build_parser().format_help()


def test_cmd_predict_depth_e2e(tmp_path, capsys):
    import csv

    from PIL import Image
    import numpy as np

    frames = tmp_path / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    arr = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(frames / "f.jpg")
    with open(frames / "keyframes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_id", "filename"])
        w.writerow([0, "f.jpg"])
    out = tmp_path / "depth"
    code = cli.main(["predict-depth", "--frames-dir", str(frames),
                     "--output-dir", str(out),
                     "--backend", "dummy", "--device", "cpu"])
    assert code == 0
    out_text = capsys.readouterr().out
    assert "RELATIVE" in out_text and "STEP 10" in out_text
