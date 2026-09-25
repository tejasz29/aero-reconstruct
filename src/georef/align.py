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
