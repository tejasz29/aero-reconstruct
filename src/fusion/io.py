"""STEP 10 — point-cloud artefact I/O (ascii PLY + index + report)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.fusion")


def save_ply(points: np.ndarray, colors: np.ndarray,
             path: str | Path) -> Path:
    """Save Nx3 float + Nx3 uint8 as ascii PLY (no open3d needed)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(pts) != len(cols):
        raise ValueError(f"points {len(pts)} != colors {len(cols)}")
    if len(pts) == 0:
        raise ValueError("cannot save an empty cloud")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(pts)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pts, cols):
            fh.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    return p


def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ascii PLY back to (points Nx3 float, colors Nx3 uint8)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"cloud not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    try:
        end = lines.index("end_header")
    except ValueError as exc:
        raise ValueError(f"bad PLY header: {p}") from exc
    data = [ln.split() for ln in lines[end + 1:] if ln.strip()]
    if not data:
        raise ValueError(f"empty PLY cloud: {p}")
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape[1] != 6:
        raise ValueError(f"PLY must have xyzrgb columns: {p}")
    return arr[:, :3], arr[:, 3:6].astype(np.uint8)
