"""STEP 9 — depth inference (preprocess -> backend -> full-res depth+conf)."""

from __future__ import annotations

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.depth")


def ensure_finite(depth: np.ndarray) -> np.ndarray:
    """Replace non-finite pixels with the finite median (never NaN downstream)."""
    d = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(d)
    if bool(finite.all()):
        return d
    if not bool(finite.any()):
        raise ValueError("depth map has no finite pixels")
    fill = float(np.median(d[finite]))
    d[~finite] = fill
    return d


def predict_single(image: np.ndarray, backend, input_size: int = 518) -> tuple[np.ndarray, np.ndarray]:
    """Full-res relative depth + confidence for one HxWx3 image.

    The backend runs at model size; its output is resized back to HxW so
    every artefact lines up with the source keyframe pixel-for-pixel.
    """
    from PIL import Image

    from src.depth.confidence import confidence_from_depth
    from src.depth.preprocess import resize_to_model

    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {img.shape}")
    h, w = img.shape[:2]
    small = resize_to_model(img, input_size=input_size)
    raw = np.asarray(backend.predict(small), dtype=np.float32)
    raw = ensure_finite(raw)
    if raw.shape != small.shape[:2]:
        pil = Image.fromarray(raw)
        raw = np.asarray(pil.resize((small.shape[1], small.shape[0]),
                                    Image.BILINEAR), dtype=np.float32)
    if (h, w) != raw.shape:
        pil = Image.fromarray(raw)
        raw = np.asarray(pil.resize((w, h), Image.BILINEAR), dtype=np.float32)
    conf = confidence_from_depth(raw)
    return raw, conf
