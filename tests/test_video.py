"""STEP 2 — video probing, sampling math, and end-to-end extraction.

End-to-end tests build a small synthetic MP4 with cv2.VideoWriter so no
real drone footage is needed.
"""

import csv

import cv2
import numpy as np
import pytest

from src import cli
from src.video.frame_extractor import (
    TIMESTAMPS_CSV,
    compute_sample_indices,
    extract_frames,
)
from src.video.video_info import frame_timestamp, probe_video


def make_synthetic_video(path, n_frames=30, fps=10, width=320, height=240):
    """Write a moving-gradient MP4 (deterministic, decodable anywhere)."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "VideoWriter failed to open"
    for i in range(n_frames):
        shade = int(255 * i / max(1, n_frames - 1))
        frame = np.full((height, width, 3), shade, dtype=np.uint8)
        cv2.rectangle(frame, (i * 3 % width, 10), (i * 3 % width + 40, 60),
                      (0, 255 - shade, shade), -1)
        writer.write(frame)
    writer.release()
    return path


# --- probing ---

def test_probe_video_reads_metadata(tmp_path):
    vid = make_synthetic_video(tmp_path / "flight.mp4")
    info = probe_video(vid)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(10.0)
    assert info.frame_count == 30
    assert info.duration_s == pytest.approx(3.0)


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "nope.mp4")


def test_frame_timestamp_math():
    assert frame_timestamp(0, 10.0) == 0.0
    assert frame_timestamp(25, 10.0) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        frame_timestamp(5, 0.0)
    with pytest.raises(ValueError):
        frame_timestamp(-1, 10.0)


# --- sampling math (pure function) ---

def test_stride_sampling():
    # 30 frames @10fps, target 2fps -> stride 5
    assert compute_sample_indices(30, 10.0, 2.0, 2000) == [0, 5, 10, 15, 20, 25]


def test_target_above_source_keeps_all():
    assert compute_sample_indices(8, 10.0, 30.0, 2000) == list(range(8))
    assert compute_sample_indices(8, 10.0, None, 2000) == list(range(8))


def test_max_frames_uniform_subsample():
    idx = compute_sample_indices(100, 10.0, 10.0, 10)
    assert len(idx) == 10
    assert idx[0] == 0 and idx[-1] == 90
    assert all(b > a for a, b in zip(idx, idx[1:]))  # order preserved


def test_empty_video_selects_nothing():
    assert compute_sample_indices(0, 10.0, 2.0, 2000) == []


# --- end-to-end extraction ---

def test_extract_frames_writes_images_and_csv(tmp_path):
    vid = make_synthetic_video(tmp_path / "flight.mp4")
    out = tmp_path / "frames"
    result = extract_frames(vid, out, target_fps=2.0, max_frames=2000,
                            max_width=1920)
    assert len(result.frames) == 6  # stride 5 over 30 frames
    assert result.csv_path == out / TIMESTAMPS_CSV
    assert result.csv_path.is_file()
    with open(result.csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    assert rows[0]["frame_id"] == "0" and rows[0]["source_index"] == "0"
    assert float(rows[2]["timestamp_s"]) == pytest.approx(1.0)  # idx 10 @10fps
    assert (out / rows[0]["filename"]).is_file()


def test_extract_frames_downscales_wide_video(tmp_path):
    vid = make_synthetic_video(tmp_path / "wide.mp4", n_frames=10,
                               width=640, height=480)
    result = extract_frames(vid, tmp_path / "frames", target_fps=10.0,
                            max_width=320)
    assert result.frames and result.frames[0].width == 320
    assert result.frames[0].height == 240
    saved = cv2.imread(str(tmp_path / "frames" / result.frames[0].filename))
    assert saved.shape[1] == 320


def test_extract_frames_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_frames(tmp_path / "nope.mp4", tmp_path / "frames")


def test_cli_extract_frames(tmp_path):
    vid = make_synthetic_video(tmp_path / "flight.mp4")
    out = tmp_path / "cli_frames"
    rc = cli.main(["extract-frames", "--video", str(vid),
                   "--frames-dir", str(out), "--target-fps", "2"])
    assert rc == 0
    assert (out / TIMESTAMPS_CSV).is_file()
    assert len(list(out.glob("frame_*.jpg"))) == 6
