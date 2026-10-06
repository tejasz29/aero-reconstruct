"""STEP 9 — depth artefact I/O (.npy maps + index CSV + report JSON)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.depth")

DEPTH_INDEX_HEADER = ["frame_id", "filename", "width", "height", "depth_path",
                      "confidence_path", "depth_min", "depth_max",
                      "depth_mean", "confidence_mean"]


def save_depth_npy(depth: np.ndarray, path: str | Path) -> Path:
    """Save a float32 HxW relative-depth map (creates parents)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.asarray(depth, dtype=np.float32))
    return p


def load_depth_npy(path: str | Path) -> np.ndarray:
    """Load a depth map; raises ValueError when missing or not 2-D."""
    arr = np.load(str(path))
    if arr.ndim != 2:
        raise ValueError(f"depth map must be HxW, got shape {arr.shape}: {path}")
    return arr.astype(np.float32)
