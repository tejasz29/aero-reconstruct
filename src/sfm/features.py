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