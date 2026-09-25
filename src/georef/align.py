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

from src.common.logging_utils import get_logger

log = get_logger("sp3d.georef.align")
