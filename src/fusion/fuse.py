"""STEP 11 — point-cloud fusion (confidence weighting, voxel downsample).

Merges STEP 10 per-frame clouds into one deduplicated scene. Pure-numpy
baseline; open3d is an optional fast path only. Metric-via-GPS-scale is
preserved; absolute CRS stays deferred to STEP 16.
"""

from __future__ import annotations

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.fusion")


def voxel_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Quantize Nx3 points to int64 voxel indices (floor-divide)."""
    if voxel_size <= 0.0 or not np.isfinite(voxel_size):
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if not bool(np.all(np.isfinite(pts))):
        raise ValueError("points must be finite for voxel hashing")
    return np.floor(pts / float(voxel_size)).astype(np.int64)


def _weighted_voxel_average(keys: np.ndarray, points: np.ndarray,
                            colors: np.ndarray,
                            confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse points sharing a voxel via confidence-weighted averages."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cols = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if not (len(keys) == len(pts) == len(cols) == len(conf)):
        raise ValueError("keys/points/colors/confidence length mismatch")
    uniq, _idx, inverse, counts = np.unique(keys, axis=0, return_index=True,
                                                return_inverse=True, return_counts=True)
    # np.unique with axis returns sorted uniq; group via inverse labels
    out_pts = np.zeros((len(uniq), 3))
    out_cols = np.zeros((len(uniq), 3))
    out_conf = np.zeros(len(uniq))
    for v in range(len(uniq)):
        mask = inverse == v
        w = np.where(np.isfinite(conf[mask]), conf[mask], 0.0)
        wsum = float(w.sum())
        if wsum <= 0.0:
            w = np.ones(mask.sum())
            wsum = float(w.sum())
        w = w / wsum
        out_pts[v] = (pts[mask] * w[:, None]).sum(axis=0)
        out_cols[v] = np.clip((cols[mask] * w[:, None]).sum(axis=0), 0, 255)
        out_conf[v] = float(conf[mask].mean()) if mask.sum() else 0.0
    return out_pts, out_cols, out_conf


def load_frame_clouds(cloud_index_csv) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load per-frame PLYs listed in ``cloud_index.csv`` (STEP 10 output)."""
    from pathlib import Path

    from src.fusion.io import load_ply, read_cloud_index

    rows = read_cloud_index(cloud_index_csv)
    if not rows:
        raise ValueError(f"no cloud rows in {cloud_index_csv}")
    base = Path(cloud_index_csv).parent
    all_pts, all_cols, all_conf, kept = [], [], [], []
    for row in rows:
        ply_rel = str(row.get("ply_path", ""))
        if not ply_rel:
            log.warning("no ply_path for %s — skipped", row.get("filename", "?"))
            continue
        ply = Path(ply_rel) if Path(ply_rel).is_absolute() else base / Path(ply_rel).name
        if not ply.is_file():
            # absolute STEP 10 paths may point at tmp dirs; try sibling name
            alt = base / Path(ply_rel).name
            ply = alt if alt.is_file() else ply
        if not ply.is_file():
            log.warning("missing frame cloud %s — skipped", ply)
            continue
        pts, cols = load_ply(ply)
        try:
            mean_conf = float(row.get("mean_confidence", 1.0) or 1.0)
        except (TypeError, ValueError):
            mean_conf = 1.0
        all_pts.append(np.asarray(pts, dtype=np.float64))
        all_cols.append(np.asarray(cols, dtype=np.uint8))
        all_conf.append(np.full(len(pts), mean_conf, dtype=np.float64))
        kept.append(row)
    if not all_pts:
        raise ValueError(f"no loadable frame clouds in {cloud_index_csv}")
    return (np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0),
            np.concatenate(all_conf, axis=0), kept)


def recover_per_point_confidence(cloud_index_csv, depth_index_csv,
                                 stride: int = 2) -> np.ndarray | None:
    """Recover per-point confidence by replaying STEP 10 valid masks.

    Returns concatenated per-point confidences aligned with
    :func:`load_frame_clouds` order, or ``None`` when depth artefacts are
    unavailable (caller falls back to frame means).
    """
    from pathlib import Path

    try:
        from src.depth.io import load_depth_npy, read_depth_index
        from src.fusion.io import read_cloud_index
        from src.fusion.unproject import valid_mask
    except ImportError:
        return None
    cloud_rows = {r.get("filename"): r for r in read_cloud_index(cloud_index_csv)}
    try:
        depth_rows = read_depth_index(depth_index_csv)
    except (FileNotFoundError, ValueError):
        return None
    base = Path(cloud_index_csv).parent
    out: list[np.ndarray] = []
    for drow in depth_rows:
        fname = str(drow.get("filename", ""))
        if fname not in cloud_rows:
            continue
        try:
            depth = load_depth_npy(drow["depth_path"])
        except (FileNotFoundError, KeyError):
            return None
        mask = valid_mask(depth)
        if stride > 1:
            keep = np.zeros_like(mask, dtype=bool)
            keep[::stride, ::stride] = True
            mask = mask & keep
        conf_path = drow.get("confidence_path", "")
        if conf_path:
            try:
                conf = np.load(str(conf_path)).astype(np.float64)[mask]
            except (FileNotFoundError, ValueError):
                conf = np.ones(int(mask.sum()), dtype=np.float64)
        else:
            conf = np.ones(int(mask.sum()), dtype=np.float64)
        out.append(conf)
    if not out:
        return None
    _ = base
    return np.concatenate(out, axis=0)


def estimate_normals_pca(points: np.ndarray, k: int = 12) -> np.ndarray:
    """PCA normals baseline: smallest eigenvector of k-nearest covariance.

    O(N^2) brute force — fine for unit tests and small scenes; the runner
    subsamples large clouds before calling this (see ``run_fusion``).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    if n == 0:
        raise ValueError("cannot estimate normals of an empty cloud")
    if k < 3:
        raise ValueError(f"k must be >= 3, got {k}")
    k = min(k, n)
    normals = np.zeros((n, 3))
    for i in range(n):
        d2 = ((pts - pts[i]) ** 2).sum(axis=1)
        nn = np.argpartition(d2, k - 1)[:k]
        cov = np.cov((pts[nn] - pts[nn].mean(axis=0)).T)
        vals, vecs = np.linalg.eigh(cov)
        normals[i] = vecs[:, 0]
    return normals


def orient_normals(points: np.ndarray, normals: np.ndarray,
                   viewpoint: np.ndarray | None = None) -> np.ndarray:
    """Flip normals toward ``viewpoint`` (default: cloud centroid +z)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if len(pts) != len(nrm):
        raise ValueError("points/normals length mismatch")
    if viewpoint is None:
        viewpoint = pts.mean(axis=0) + np.array([0.0, 0.0, 1.0])
    vp = np.asarray(viewpoint, dtype=np.float64).reshape(3)
    to_view = vp - pts
    flip = (nrm * to_view).sum(axis=1) < 0.0
    nrm[flip] *= -1.0
    return nrm


def estimate_normals(points: np.ndarray, k: int = 12,
                     viewpoint: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """Estimate normals, preferring open3d when installed.

    Returns ``(normals, backend)`` where backend is ``"open3d"`` or
    ``"pca"``. Never raises for a missing open3d — always falls back.
    """
    try:
        import open3d as o3d  # type: ignore

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            np.asarray(points, dtype=np.float64)))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max(3, k)))
        nrm = np.asarray(pcd.normals)
        if len(nrm) != len(np.asarray(points).reshape(-1, 3)):
            raise RuntimeError("open3d normal count mismatch")
        return orient_normals(points, nrm, viewpoint), "open3d"
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 — open3d failure falls back
        log.warning("open3d normals failed (%s) — using PCA baseline", exc)
    return orient_normals(points, estimate_normals_pca(points, k), viewpoint), "pca"


def voxel_downsample(points: np.ndarray, colors: np.ndarray,
                     confidence: np.ndarray | None = None,
                     voxel_size: float = 0.10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample a cloud to one point per voxel (weighted average)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        raise ValueError("cannot downsample an empty cloud")
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(cols) != len(pts):
        raise ValueError(f"points {len(pts)} != colors {len(cols)}")
    conf = (np.ones(len(pts), dtype=np.float64) if confidence is None
            else np.asarray(confidence, dtype=np.float64).reshape(-1))
    if len(conf) != len(pts):
        raise ValueError("confidence length mismatch")
    keys = voxel_keys(pts, voxel_size)
    fused_pts, fused_cols, fused_conf = _weighted_voxel_average(keys, pts, cols, conf)
    return fused_pts, np.clip(fused_cols, 0, 255).astype(np.uint8), fused_conf


def fuse_clouds(points: np.ndarray, colors: np.ndarray,
                confidence: np.ndarray | None = None,
                voxel_size: float = 0.10,
                estimate_normals_flag: bool = True,
                normals_k: int = 12) -> dict:
    """Full STEP 11 fusion: voxel-dedupe then optional normals.

    Returns dict with ``points/colors/confidence/normals`` (normals None
    when disabled), ``n_in/n_out``, ``normals_backend`` and ``kept_ratio``.
    Never invents geometry — output is a subset-averaging of the input.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n_in = len(pts)
    if n_in == 0:
        raise ValueError("cannot fuse an empty cloud")
    fused_pts, fused_cols, fused_conf = voxel_downsample(
        pts, colors, confidence, voxel_size)
    normals = None
    backend = "none"
    if estimate_normals_flag and len(fused_pts) >= 3:
        normals, backend = estimate_normals(fused_pts, k=normals_k)
    elif estimate_normals_flag:
        log.warning("too few fused points (%d) for normals — skipped", len(fused_pts))
    return {"points": fused_pts, "colors": fused_cols, "confidence": fused_conf,
            "normals": normals, "normals_backend": backend,
            "n_in": n_in, "n_out": len(fused_pts),
            "kept_ratio": len(fused_pts) / max(1, n_in)}
