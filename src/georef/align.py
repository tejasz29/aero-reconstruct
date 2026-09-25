"""STEP 8 — Visual <-> GPS similarity alignment.

Resolves the monocular scale ambiguity from STEP 5: the SfM trajectory
lives in an arbitrary relative frame (world = first keyframe, unit-less
baseline), while STEP 7 provides metric GPS positions in ENU/UTM.

We fit ``X_global ~= s * R * X_visual + t`` with a robust Umeyama +
RANSAC estimator, then apply it to all kept camera centres (and rotate
their world->camera rotations accordingly). RTK/PPK fixes, when present,
are honoured via ``alignment.use_rtk_if_available`` (tighter inliers).
"""

from __future__ import annotations

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.georef.align")


class SimilarityTransform:
    """Scale-rotation-translation map: X_global = s * R @ X_visual + t."""

    def __init__(self, scale: float = 1.0,
                 rotation: np.ndarray | None = None,
                 translation: np.ndarray | None = None):
        self.scale = float(scale)
        self.R = np.eye(3) if rotation is None else np.asarray(
            rotation, dtype=np.float64).reshape(3, 3)
        self.t = np.zeros(3) if translation is None else np.asarray(
            translation, dtype=np.float64).reshape(3)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (f"SimilarityTransform(scale={self.scale:.6f}, "
                f"t={self.t.tolist()})")


class VisualSample:
    """One kept SfM camera centre with its timestamp."""

    def __init__(self, timestamp_s: float, position: np.ndarray,
                 frame_id: int = 0, filename: str = ""):
        self.timestamp_s = float(timestamp_s)
        self.position = np.asarray(position, dtype=np.float64).reshape(3)
        self.frame_id = int(frame_id)
        self.filename = str(filename)


class MetricSample:
    """One GPS metric position (easting, northing, up) with timestamp."""

    def __init__(self, timestamp_s: float, position: np.ndarray):
        self.timestamp_s = float(timestamp_s)
        self.position = np.asarray(position, dtype=np.float64).reshape(3)


class AlignmentResult:
    """Everything produced by one :func:`run_alignment` call."""

    def __init__(self, transform: SimilarityTransform,
                 n_correspondences: int = 0,
                 n_inliers: int = 0,
                 rmse_inliers_m: float | None = None,
                 rmse_all_m: float | None = None,
                 inlier_ratio: float = 0.0,
                 crs: str = "", zone: str | None = None,
                 aligned_csv: str = "", transform_json: str = "",
                 report_json: str = ""):
        self.transform = transform
        self.n_correspondences = int(n_correspondences)
        self.n_inliers = int(n_inliers)
        self.rmse_inliers_m = rmse_inliers_m
        self.rmse_all_m = rmse_all_m
        self.inlier_ratio = float(inlier_ratio)
        self.crs = str(crs)
        self.zone = zone
        self.aligned_csv = str(aligned_csv)
        self.transform_json = str(transform_json)
        self.report_json = str(report_json)


def estimate_similarity_umeyama(src: np.ndarray, dst: np.ndarray,
                                with_scale: bool = True) -> SimilarityTransform:
    """Least-squares similarity ``dst ~= s * R @ src + t`` (Umeyama 1991).

    Args:
        src: (N, 3) visual positions.  dst: (N, 3) metric positions.
    Raises:
        ValueError: fewer than 3 points or degenerate (zero-variance) input.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError(
            f"need >=3 correspondences with matching shapes, got "
            f"{src.shape} vs {dst.shape}")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = float((src_c ** 2).sum() / len(src))
    if var_src < 1e-12:
        raise ValueError("degenerate visual configuration: zero variance")
    cov = (dst_c.T @ src_c) / len(src)
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1.0
    R = u @ s_mat @ vt
    if with_scale:
        scale = float((d * np.diag(s_mat)).sum() / var_src)
    else:
        scale = 1.0
    t = mu_dst - scale * R @ mu_src
    return SimilarityTransform(scale=scale, rotation=R, translation=t)


def apply_similarity(points: np.ndarray,
                     transform: SimilarityTransform) -> np.ndarray:
    """Map visual points into the global frame: ``s * R @ X + t``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return transform.scale * (pts @ transform.R.T) + transform.t


def compute_residuals_m(src: np.ndarray, dst: np.ndarray,
                        transform: SimilarityTransform) -> np.ndarray:
    """Per-correspondence Euclidean error in metres."""
    pred = apply_similarity(src, transform)
    return np.linalg.norm(np.asarray(dst, dtype=np.float64) - pred, axis=1)


def rmse_m(errors: np.ndarray) -> float | None:
    """Root-mean-square of per-point errors; None when empty."""
    err = np.asarray(errors, dtype=np.float64).ravel()
    return float(np.sqrt((err ** 2).mean())) if err.size else None


def is_valid_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    """Check orthonormality (R @ R.T ~= I) and det(R) ~= +1."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return False
    if abs(float(np.linalg.det(R)) - 1.0) > 1e-4:
        return False
    return bool(np.allclose(R @ R.T, np.eye(3), atol=tol))


def read_visual_trajectory(poses_csv) -> list[VisualSample]:
    """Parse kept STEP 5 poses into timestamped camera centres."""
    import csv
    from pathlib import Path

    path = Path(poses_csv)
    if not path.is_file():
        raise FileNotFoundError(
            f"camera poses not found: {path} — run reconstruct-poses first")
    samples: list[VisualSample] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("kept", "1").strip() not in ("1", "True", "true"):
                continue
            try:
                pos = np.array([float(row["tx"]), float(row["ty"]),
                                float(row["tz"])], dtype=np.float64)
            except (KeyError, ValueError):
                continue
            samples.append(VisualSample(
                timestamp_s=float(row["timestamp_s"]), position=pos,
                frame_id=int(row.get("frame_id") or 0),
                filename=str(row.get("filename") or "")))
    samples.sort(key=lambda s: s.timestamp_s)
    if not samples:
        raise ValueError(f"no kept poses in {path}")
    return samples


def read_metric_trajectory(gps_metric_csv) -> tuple[list[MetricSample], str, str | None]:
    """Parse STEP 7 ``gps_metric.csv`` into timestamped metric positions."""
    import csv
    from pathlib import Path

    path = Path(gps_metric_csv)
    if not path.is_file():
        raise FileNotFoundError(
            f"gps metric file not found: {path} — run convert-gps first")
    samples: list[MetricSample] = []
    crs, zone = "enu", None
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pos = np.array([float(row["easting_m"]), float(row["northing_m"]),
                            float(row["up_m"])], dtype=np.float64)
            samples.append(MetricSample(float(row["timestamp_s"]), pos))
            crs = str(row.get("crs") or crs)
            zone = str(row.get("zone") or "") or None
    samples.sort(key=lambda s: s.timestamp_s)
    if not samples:
        raise ValueError(f"no metric fixes in {path}")
    return samples, crs, zone


def interpolate_metric_to_visual(
        visual: list[VisualSample],
        metric: list[MetricSample]) -> tuple[np.ndarray, np.ndarray, list[VisualSample]]:
    """Linearly interpolate GPS metric positions onto visual timestamps.

    Visual timestamps outside the GPS span are dropped (no extrapolation —
    claiming scale where no GPS exists would be dishonest). Returns
    ``(src, dst, kept_visual)`` with (N, 3) arrays.
    """
    if not visual or not metric:
        raise ValueError("need non-empty visual and metric trajectories")
    g_times = np.array([m.timestamp_s for m in metric], dtype=np.float64)
    g_pos = np.array([m.position for m in metric], dtype=np.float64)
    order = np.argsort(g_times)
    g_times, g_pos = g_times[order], g_pos[order]
    src_list, dst_list, kept = [], [], []
    for v in visual:
        if not (g_times[0] <= v.timestamp_s <= g_times[-1]):
            continue
        interp = np.array([np.interp(v.timestamp_s, g_times, g_pos[:, k])
                           for k in range(3)], dtype=np.float64)
        src_list.append(v.position)
        dst_list.append(interp)
        kept.append(v)
    if not kept:
        raise ValueError("no timestamp overlap between visual and GPS trajectories")
    return np.asarray(src_list), np.asarray(dst_list), kept


def build_correspondences(poses_csv, gps_metric_csv,
                          min_correspondences: int = 6
                          ) -> tuple[np.ndarray, np.ndarray, list[VisualSample], str, str | None]:
    """Full STEP 5 + STEP 7 -> (src, dst) correspondence builder.

    Raises ``ValueError`` when fewer than ``min_correspondences`` pairs
    survive timestamp overlap.
    """
    visual = read_visual_trajectory(poses_csv)
    metric, crs, zone = read_metric_trajectory(gps_metric_csv)
    src, dst, kept = interpolate_metric_to_visual(visual, metric)
    if len(kept) < min_correspondences:
        raise ValueError(
            f"only {len(kept)} visual<->GPS correspondences "
            f"(need >= {min_correspondences}) — check timestamp overlap")
    return src, dst, kept, crs, zone
