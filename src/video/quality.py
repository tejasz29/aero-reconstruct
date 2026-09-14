"""STEP 3 — per-frame quality scoring.

Three cheap, well-understood signals (all computed on grayscale):

* **blur** — variance of the Laplacian; low values mean a blurry frame.
* **exposure** — mean gray value; outside [min, max] means too dark /
  saturated to match or texture from.
* **features** — ORB (default) or SIFT keypoint count; frames with almost
  nothing to track starve SfM (STEP 5).

Plus **dHash** (difference hash) so the selector can drop near-duplicate
frames. All functions are pure (numpy arrays in, numbers out) except
``load_gray``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.video")


def load_gray(path: str | Path) -> np.ndarray:
    """Load an image as grayscale; raises ValueError when unreadable."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return img


def blur_score(gray: np.ndarray) -> float:
    """Laplacian variance — higher means sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_mean(gray: np.ndarray) -> float:
    """Mean gray value in [0, 255]."""
    return float(gray.mean())


def count_features(gray: np.ndarray, detector: str = "orb",
                   nfeatures: int = 2000) -> int:
    """Number of detectable keypoints (ORB default; SIFT on request)."""
    name = detector.lower()
    if name == "sift":
        try:
            det = cv2.SIFT_create(nfeatures)
        except (AttributeError, cv2.error) as exc:
            log.warning("SIFT unavailable (%s) — falling back to ORB", exc)
            det = cv2.ORB_create(nfeatures)
    elif name == "orb":
        det = cv2.ORB_create(nfeatures)
    else:
        raise ValueError(f"unknown feature detector: {detector!r} (orb|sift)")
    keypoints = det.detect(gray, None)
    return len(keypoints)


def dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    """64-bit difference hash as an int (perceptual, translation-tolerant)."""
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = np.packbits(diff.flatten())
    return int.from_bytes(bits.tobytes(), "big")


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(a ^ b).count("1")
