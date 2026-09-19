"""STEP 5 — feature detection, description and matching.

Classical-SfM front end used by the OpenCV tracker:

* ``extract_features`` — SIFT (default) or ORB keypoints/descriptors,
  soft-capped by ``max_features``.
* ``match_features`` — brute-force kNN matching with Lowe's ratio test.

The caller feeds keyframe images; everything downstream (essential matrix,
chaining) works purely on the 2D point pairs returned here.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.sfm")


def extract_features(
    gray: np.ndarray,
    feature: str = "sift",
    max_features: int = 8000,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Detect and describe keypoints in a grayscale image.

    Returns ``(keypoints, descriptors)``; ``descriptors`` is ``None`` when
    the image carries no extractable structure (caller rejects the frame).
    """
    feature = feature.lower()
    if gray is None or gray.size == 0:
        return [], None
    if feature == "sift":
        detector = cv2.SIFT_create(nfeatures=max_features)
    elif feature == "orb":
        detector = cv2.ORB_create(nfeatures=max_features)
    else:
        raise ValueError(f"unknown feature detector: {feature!r} (sift|orb)")
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is not None:
        keypoints = list(keypoints)[:len(descriptors)]
    return keypoints, descriptors


def match_features(
    des1: np.ndarray | None,
    des2: np.ndarray | None,
    ratio_test: float = 0.8,
    hamming: bool = False,
    cross_check: bool = True,
) -> np.ndarray:
    """Match two descriptor sets; returns (N, 2) int array of index pairs.

    Brute-force kNN (k=2) with Lowe's ratio test, plus reciprocal-consistency
    (a pair survives only if each is each other's best candidate). Empty
    input returns a (0, 2) array so callers never special-case ``None``.
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.empty((0, 2), dtype=np.int64)
    norm = cv2.NORM_HAMMING if hamming else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)

    def ratio_pairs(d1, d2):
        out = set()
        for pair in matcher.knnMatch(d1, d2, k=2):
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < ratio_test * second.distance:
                out.add((best.queryIdx, best.trainIdx))
        return out

    forward = ratio_pairs(des1, des2)
    if not forward:
        return np.empty((0, 2), dtype=np.int64)
    if not cross_check:
        pairs = sorted(forward)
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    reverse = ratio_pairs(des2, des1)
    pairs = sorted((a, b) for a, b in forward if (b, a) in reverse)
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)