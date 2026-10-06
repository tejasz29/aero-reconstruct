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
