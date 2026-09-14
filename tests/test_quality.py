"""STEP 3 — quality scoring and keyframe selection."""

import csv

import cv2
import numpy as np
import pytest

from src import cli
from src.video.frame_extractor import TIMESTAMPS_CSV, read_timestamps_csv
from src.video.keyframes import (
    KEYFRAMES_CSV,
    SCORES_CSV,
    FrameScore,
    run_selection,
    select_keyframes,
)
from src.video.quality import (
    blur_score,
    count_features,
    dhash,
    exposure_mean,
    hamming_distance,
    load_gray,
)


def checkerboard(squares=8, size=30):
    pattern = np.indices((squares, squares)).sum(axis=0) % 2 * 255
    return np.kron(pattern, np.ones((size, size))).astype(np.uint8)


def test_blur_sharp_beats_blurred():
    sharp = checkerboard()
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
    assert blur_score(sharp) > 10 * blur_score(blurred)


def test_exposure_mean_values():
    assert exposure_mean(np.zeros((10, 10), np.uint8)) == 0.0
    assert exposure_mean(np.full((10, 10), 255, np.uint8)) == 255.0
    assert exposure_mean(np.full((10, 10), 128, np.uint8)) == pytest.approx(128.0)


def test_feature_count_textured_vs_blank():
    assert count_features(checkerboard()) > 50
    assert count_features(np.full((240, 240), 128, np.uint8)) == 0


def test_feature_count_unknown_detector_raises():
    with pytest.raises(ValueError):
        count_features(checkerboard(), detector="bogus")


def test_load_gray_missing_raises(tmp_path):
    with pytest.raises(ValueError):
        load_gray(tmp_path / "nope.jpg")


def test_dhash_identical_is_zero_and_stable():
    img = checkerboard()
    assert hamming_distance(dhash(img), dhash(img)) == 0
    other = np.roll(img, 40, axis=1)
    assert hamming_distance(dhash(img), dhash(other)) > 0


def _row(frame_id, ts, blur=1000.0, exposure=120.0, features=500):
    return FrameScore(frame_id=frame_id, source_index=frame_id,
                      timestamp_s=ts, filename=f"frame_{frame_id:06d}.jpg",
                      blur=blur, exposure=exposure, features=features)


def _select(rows, hashes=None, **kw):
    defaults = dict(blur_threshold=100.0, exposure_min=15.0,
                    exposure_max=240.0, min_features=300,
                    min_time_gap_s=0.0, dedup_hamming_threshold=-1,
                    max_keep=None)
    defaults.update(kw)
    return select_keyframes(rows, hashes or {}, **defaults)


def test_gates_reject_bad_frames():
    rows = [ _row(0, 0.0),                       # good
             _row(1, 1.0, blur=5.0),             # blurry
             _row(2, 2.0, exposure=5.0),         # too dark
             _row(3, 3.0, exposure=250.0),       # saturated
             _row(4, 4.0, features=10) ]         # featureless
    out = _select(rows)
    verdict = {r.frame_id: (r.kept, r.reject_reason) for r in out}
    assert verdict[0] == (True, "")
    assert verdict[1] == (False, "blur")
    assert verdict[2] == (False, "exposure")
    assert verdict[3] == (False, "exposure")
    assert verdict[4] == (False, "features")


def test_time_gap_enforced():
    rows = [_row(0, 0.0), _row(1, 0.1), _row(2, 1.0)]
    out = _select(rows, min_time_gap_s=0.5)
    verdict = {r.frame_id: (r.kept, r.reject_reason) for r in out}
    assert verdict[0][0] and verdict[2][0]
    assert verdict[1] == (False, "time_gap")


def test_duplicates_rejected_by_hash():
    rows = [_row(0, 0.0), _row(1, 1.0), _row(2, 2.0)]
    hashes = {0: 0b0000, 1: 0b0001, 2: 0b11111111}  # 1 differs by 1 bit
    out = _select(rows, hashes, dedup_hamming_threshold=5)
    verdict = {r.frame_id: (r.kept, r.reject_reason) for r in out}
    assert verdict[0][0] and verdict[2][0]
    assert verdict[1] == (False, "duplicate")


def test_max_keep_uniform_subsample():
    rows = [_row(i, float(i)) for i in range(6)]
    out = _select(rows, max_keep=2)
    kept = [r.frame_id for r in out if r.kept]
    assert kept == [0, 3]  # uniform coverage, not just the first two
