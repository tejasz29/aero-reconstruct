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
