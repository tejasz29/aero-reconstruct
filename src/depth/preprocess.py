"""STEP 9 — image preprocessing for monocular depth.

PIL-first on purpose: the depth stage must stay importable without the
heavy video stack (cv2) so unit tests and the CPU fallback run anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.depth")


def load_image(path: str | Path) -> np.ndarray:
    """Load an RGB image as uint8 HxWx3; raises ValueError when unreadable."""
    from PIL import Image

    p = Path(path)
    try:
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"cannot read image: {path} ({exc})") from exc


def resize_to_model(image: np.ndarray, input_size: int = 518) -> np.ndarray:
    """Resize so the longest side equals ``input_size`` (aspect preserved).

    Depth Anything family uses ~518 px; smaller is faster but blurrier.
    Never upscales tiny thumbnails by more than 2x to avoid fake detail.
    """
    from PIL import Image

    if input_size <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}")
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest == input_size:
        return image
    scale = min(input_size / longest, 2.0)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    pil = Image.fromarray(image)
    return np.asarray(pil.resize((new_w, new_h), Image.BILINEAR))
