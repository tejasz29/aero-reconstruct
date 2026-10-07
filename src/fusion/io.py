"""STEP 10 — point-cloud artefact I/O (ascii PLY + index + report)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.fusion")


def save_ply(points: np.ndarray, colors: np.ndarray,
             path: str | Path, normals: np.ndarray | None = None) -> Path:
    """Save Nx3 float + Nx3 uint8 as ascii PLY (no open3d needed).

    ``normals`` is opt-in (STEP 11): when given, ``nx/ny/nz`` float
    properties are appended; the 6-column reader stays backward compatible.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(pts) != len(cols):
        raise ValueError(f"points {len(pts)} != colors {len(cols)}")
    if len(pts) == 0:
        raise ValueError("cannot save an empty cloud")
    nrm = None
    if normals is not None:
        nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        if len(nrm) != len(pts):
            raise ValueError(f"points {len(pts)} != normals {len(nrm)}")
        if not bool(np.all(np.isfinite(nrm))):
            raise ValueError("normals must be finite")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(pts)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if nrm is not None:
            fh.write("property float nx\nproperty float ny\nproperty float nz\n")
        fh.write("end_header\n")
        if nrm is None:
            for (x, y, z), (r, g, b) in zip(pts, cols):
                fh.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for (x, y, z), (r, g, b), (nx, ny, nz) in zip(pts, cols, nrm):
                fh.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} "
                         f"{nx:.6f} {ny:.6f} {nz:.6f}\n")
    return p


def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ascii PLY back to (points Nx3 float, colors Nx3 uint8).

    Accepts 6-column (xyzrgb) and 9-column (xyzrgb+nxnynz) clouds; normals
    are ignored here — use :func:`load_ply_with_normals` when needed.
    """
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
    if arr.shape[1] not in (6, 9):
        raise ValueError(f"PLY must have xyzrgb[+normals] columns: {p}")
    return arr[:, :3], arr[:, 3:6].astype(np.uint8)


def load_ply_with_normals(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load a PLY cloud, returning normals too (None for 6-column files)."""
    pts, cols = load_ply(path)
    p = Path(path)
    with open(p, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    end = lines.index("end_header")
    data = [ln.split() for ln in lines[end + 1:] if ln.strip()]
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape[1] == 9:
        return pts, cols, arr[:, 6:9]
    return pts, cols, None


def validate_cloud(points: np.ndarray, colors: np.ndarray,
                     normals: np.ndarray | None = None) -> None:
    """Raise when a cloud breaks the finite/shape/color contract."""
    pts = np.asarray(points)
    cols = np.asarray(colors)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got shape {pts.shape}")
    if cols.shape != pts.shape:
        raise ValueError(f"colors {cols.shape} != points {pts.shape}")
    if not bool(np.all(np.isfinite(pts))):
        raise ValueError("cloud points must be finite")
    if cols.min() < 0 or cols.max() > 255:
        raise ValueError("cloud colors must be in [0, 255]")
    if normals is not None:
        nrm = np.asarray(normals)
        if nrm.shape != pts.shape:
            raise ValueError(f"normals {nrm.shape} != points {pts.shape}")
        if not bool(np.all(np.isfinite(nrm))):
            raise ValueError("cloud normals must be finite")


CLOUD_INDEX_HEADER = ["frame_id", "filename", "n_points", "ply_path",
                      "mean_confidence"]


def write_cloud_index(rows: list[dict], path: str | Path) -> Path:
    """Write per-frame cloud audit CSV (input to STEP 11 fusion)."""
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLOUD_INDEX_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CLOUD_INDEX_HEADER})
    log.info("cloud index written: %s (%d rows)", p, len(rows))
    return p


def read_cloud_index(path: str | Path) -> list[dict]:
    """Read the per-frame cloud index back."""
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_unproject_report(report: dict, path: str | Path) -> Path:
    """Write ``unproject_report.json`` (scale note + stats + paths)."""
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    log.info("unproject report written: %s", p)
    return p
