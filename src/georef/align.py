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
