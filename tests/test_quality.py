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
