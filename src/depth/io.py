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
    try:
        arr = np.load(str(path))
    except FileNotFoundError as exc:
        raise ValueError(f"depth map not found: {path}") from exc
    if arr.ndim != 2:
        raise ValueError(f"depth map must be HxW, got shape {arr.shape}: {path}")
    return arr.astype(np.float32)


def save_preview_png(depth: np.ndarray, path: str | Path) -> Path:
    """Human-viewable grayscale preview (relative depth normalised to 0-255)."""
    from PIL import Image

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = np.asarray(depth, dtype=np.float64)
    lo, hi = float(np.min(d)), float(np.max(d))
    span = hi - lo if hi > lo else 1.0
    norm = ((d - lo) / span * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(norm, mode="L").save(p)
    return p


def save_confidence_npy(conf: np.ndarray, path: str | Path) -> Path:
    """Save a float32 HxW confidence map in [0, 1] (validates first)."""
    from src.depth.confidence import validate_confidence

    c = np.asarray(conf, dtype=np.float32)
    validate_confidence(c, c.shape)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, c)
    return p


def write_depth_index(rows: list[dict], path: str | Path) -> Path:
    """Write ``depth_index.csv`` (one row per keyframe, auditable)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DEPTH_INDEX_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in DEPTH_INDEX_HEADER})
    log.info("depth index written: %s (%d rows)", p, len(rows))
    return p


def read_depth_index(path: str | Path) -> list[dict]:
    """Read ``depth_index.csv`` back (used by STEP 10 and tests)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_depth_report(report: dict, path: str | Path) -> Path:
    """Write ``depth_report.json`` (config, counts, stats, relative note)."""
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    log.info("depth report written: %s", p)
    return p
