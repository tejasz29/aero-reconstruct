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


def identity_transform() -> SimilarityTransform:
    """Unit scale, identity rotation, zero offset — the no-op alignment."""
    return SimilarityTransform(1.0, np.eye(3), np.zeros(3))


def apply_similarity(transform: SimilarityTransform,
                    points: np.ndarray) -> np.ndarray:
    """Map visual-frame points into the global metric frame, as ``(N, 3)``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts
    R = np.asarray(transform.R, dtype=np.float64)
    return (transform.scale * (pts @ R.T)
            + np.asarray(transform.t, dtype=np.float64))


def apply_similarity_to_rotation(transform: SimilarityTransform,
                                 R_wc: np.ndarray) -> np.ndarray:
    """World->camera rotation re-expressed in the global frame: ``R_wc @ R.T``.

    The similarity scale deliberately does not appear. After the alignment the
    scene points *and* the camera centres are metric, so the projection
    ``X_cam = R_gc @ (X_global - C_global)`` already yields metres and can be
    used directly with the intrinsics and the learned depth of STEP 9/10.
    """
    R = np.asarray(transform.R, dtype=np.float64)
    return np.asarray(R_wc, dtype=np.float64).reshape(3, 3) @ R.T


def invert_similarity(transform: SimilarityTransform) -> SimilarityTransform:
    """Inverse transform, mapping the global metric frame back to the visual one.

    Needed downstream to push a metric point (e.g. a GPS-referenced control
    point) into the reconstruction's own frame, or to re-project a global
    camera centre onto the STEP 6 plots.
    """
    R = np.asarray(transform.R, dtype=np.float64)
    R_inv = R.T
    scale_inv = 1.0 / transform.scale
    t_inv = -scale_inv * (R_inv @ np.asarray(transform.t, dtype=np.float64))
    return SimilarityTransform(scale_inv, R_inv, t_inv)


def transform_to_dict(transform: SimilarityTransform) -> dict:
    """JSON-ready dict of the transform, with the model written out explicitly.

    The ``model`` field is stored rather than implied so that a consumer of
    ``alignment_transform.json`` (STEP 10, STEP 16, the viewer) never has to
    guess whether the translation is applied before or after the rotation.
    """
    R = np.asarray(transform.R, dtype=np.float64)
    t = np.asarray(transform.t, dtype=np.float64)
    return {
        "model": "X_global = scale * (R @ X_visual) + translation_m",
        "scale": float(transform.scale),
        "rotation": [[float(v) for v in row] for row in R],
        "translation_m": [float(v) for v in t],
    }


def transform_from_dict(data: dict) -> SimilarityTransform:
    """Rebuild a :class:`SimilarityTransform` from :func:`transform_to_dict`."""
    if not isinstance(data, dict):
        raise ValueError("transform payload must be a dict")
    for key in ("scale", "rotation", "translation_m"):
        if key not in data:
            raise ValueError(f"transform payload is missing '{key}'")
    rotation = np.asarray(data["rotation"], dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(
            f"transform rotation must be 3x3, got {rotation.shape}")
    translation = np.asarray(data["translation_m"], dtype=np.float64).reshape(-1)
    if translation.shape != (3,):
        raise ValueError(
            f"transform translation must have 3 components, got {translation.shape}")
    return SimilarityTransform(float(data["scale"]), rotation, translation)


@dataclass
class AlignmentFit:
    """Outcome of one robust similarity fit over a set of correspondences."""

    transform: SimilarityTransform
    n_correspondences: int
    inlier_mask: np.ndarray
    n_inliers: int
    rmse_m: float
    median_residual_m: float
    max_residual_m: float
    iterations: int = 0
    success: bool = True
    reject_reason: str = ""

    @property
    def inlier_ratio(self) -> float:
        """Fraction of correspondences that survived the residual gate."""
        if not self.n_correspondences:
            return 0.0
        return self.n_inliers / self.n_correspondences

    @property
    def outlier_indices(self) -> list[int]:
        """Indices of the rejected correspondences (audit trail)."""
        mask = np.asarray(self.inlier_mask, dtype=bool).reshape(-1)
        return [int(i) for i in np.flatnonzero(~mask)]


@dataclass
class AlignmentResult:
    """Everything produced by one :func:`run_alignment` call."""

    transform: SimilarityTransform
    crs: str
    mode: str
    n_correspondences: int
    n_inliers: int
    rmse_m: float
    median_residual_m: float
    max_residual_m: float
    rtk: "RtkInfo"
    accuracy: dict
    rows: list[dict]
    zone: str | None = None
    success: bool = True
    reject_reason: str = ""
    aligned_csv: str = ""
    transform_json: str = ""
    report_json: str = ""

    @property
    def inlier_ratio(self) -> float:
        """Fraction of correspondences that survived the residual gate."""
        if not self.n_correspondences:
            return 0.0
        return self.n_inliers / self.n_correspondences

    @property
    def scale(self) -> float:
        """Metres per visual (SfM) unit — the resolved monocular scale."""
        return self.transform.scale
