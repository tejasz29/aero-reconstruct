"""STEP 8 — visual <-> GPS similarity alignment (monocular scale resolution).

A monocular SfM reconstruction (STEP 5) is correct only up to one unknown
global scale, and its world frame is an arbitrary first-keyframe frame. This
module fits the single similarity transform

    X_global = s * (R @ X_visual) + t

that maps the estimated camera centres onto the STEP 7 metric GPS frame,
fixing both the absolute scale and the global orientation of the whole
reconstruction (terrain, buildings, roads) rather than of one frame.

Method, in order:

1. pair accepted camera poses with GPS fixes by timestamp
   (:func:`associate_by_timestamp`),
2. reject outliers — GPS spikes, unmatched poses, timing jitter — with RANSAC
   on the residual distance (:func:`ransac_similarity`),
3. re-fit the inliers in closed form (Umeyama) and report the residual RMSE.

Accuracy policy (non-negotiable here, as everywhere in this project): the
reported RMSE describes *how well the trajectory was aligned to the GPS
track*. It is not a claim of the reconstruction's absolute accuracy, and it is
not centimetre-level unless the flight log actually carries RTK/PPK fixes —
ordinary consumer GPS is metre-level. :func:`accuracy_summary` states which of
the two applies instead of letting the reader guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.georef")

#: Correspondences needed to constrain a 7-dof similarity transform.
MIN_SIMILARITY_SAMPLES = 3

#: 3D alignment modes accepted by the fitters.
ALIGNMENT_MODES = ("3d", "2d")


@dataclass(frozen=True)
class SimilarityTransform:
    """The visual -> global mapping ``X_global = scale * (R @ X_visual) + t``.

    ``scale`` is positive and finite; ``R`` is a 3x3 rotation (orthonormal,
    determinant +1 — a mirror is not a valid camera-frame change of basis);
    ``t`` is the 3-vector offset in the global metric frame, in metres.
    """

    scale: float
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        R = np.asarray(self.R, dtype=np.float64)
        t = np.asarray(self.t, dtype=np.float64).reshape(-1)
        if R.shape != (3, 3):
            raise ValueError(f"rotation must be 3x3, got {R.shape}")
        if t.shape != (3,):
            raise ValueError(f"translation must have 3 components, got {t.shape}")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(
                f"scale must be positive and finite, got {self.scale}")
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            raise ValueError("rotation and translation must be finite")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)
