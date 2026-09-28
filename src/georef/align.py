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


def _as_point_arrays(source: np.ndarray, target: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Validate and reshape a correspondence pair to two ``(N, 3)`` arrays."""
    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] != dst.shape[0]:
        raise ValueError(
            f"source/target length mismatch: {src.shape[0]} vs {dst.shape[0]}")
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        raise ValueError("correspondence points must all be finite")
    return src, dst


def similarity_residuals(transform: SimilarityTransform, source: np.ndarray,
                         target: np.ndarray) -> np.ndarray:
    """Per-correspondence residual distance in metres after applying ``transform``.

    The gate used by :func:`ransac_similarity` is exactly this number: a pair
    whose mapped source lands further than the threshold from its target is an
    outlier, which is how GPS spikes and mismatched poses get thrown out.
    """
    src, dst = _as_point_arrays(source, target)
    if src.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(apply_similarity(transform, src) - dst, axis=1)


def residual_stats(residuals: np.ndarray) -> tuple[float, float, float]:
    """``(rmse, median, max)`` residual distance, all in metres.

    An empty residual vector (no correspondences at all) reports zeros rather
    than NaN so the report stays JSON-clean.
    """
    res = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if res.size == 0:
        return 0.0, 0.0, 0.0
    return (float(np.sqrt(np.mean(res ** 2))),
            float(np.median(res)),
            float(np.max(res)))


def _centroids(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split points into their centroid and the centred coordinates."""
    mu = points.mean(axis=0)
    return mu, points - mu


def estimate_similarity(source: np.ndarray, target: np.ndarray,
                        with_scale: bool = True) -> SimilarityTransform:
    """Closed-form least-squares similarity fit (Umeyama, 1991).

    Minimises ``sum_i || target_i - (s R source_i + t) ||^2`` over ``s``, the
    rotation ``R`` and the offset ``t``. With ``with_scale=False`` the rotation
    stays rigid and the scale is pinned to 1 — only meaningful when the scale
    is already known from another source.

    Requires at least :data:`MIN_SIMILARITY_SAMPLES` correspondences whose
    source points actually span 3D. A coplanar but non-degenerate path (the
    normal case for a drone pass) is fine; coincident or collinear sources
    cannot constrain a 3D similarity and raise ``ValueError``.
    """
    src, dst = _as_point_arrays(source, target)
    n = src.shape[0]
    if n < MIN_SIMILARITY_SAMPLES:
        raise ValueError(
            f"a 3D similarity needs at least {MIN_SIMILARITY_SAMPLES} "
            f"correspondences, got {n}")
    mu_s, centred_s = _centroids(src)
    mu_d, centred_d = _centroids(dst)
    covariance = (centred_d.T @ centred_s) / n
    U, singular, Vt = np.linalg.svd(covariance)
    # det(U Vt) == -1 means the data are mirrored: a camera-frame change of
    # basis cannot mirror, so fold the reflection into the smallest singular
    # value (the classic Umeyama/ Kabsch correction).
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    R = U @ correction @ Vt
    if not with_scale:
        return SimilarityTransform(1.0, R, mu_d - R @ mu_s)

    variance = float((centred_s ** 2).sum() / n)
    if variance <= 1e-18:
        raise ValueError("degenerate source points: zero spatial variance")
    scale = float(np.trace(correction @ np.diag(singular)) / variance)
    if scale <= 0.0:
        raise ValueError(
            f"fitted scale is not positive ({scale:.6g}) — the source and "
            "target sets are inconsistent (mirrored?)")
    return SimilarityTransform(scale, R, mu_d - scale * (R @ mu_s))


def _similarity_2d(source_xy: np.ndarray,
                   target_xy: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Closed-form 2D similarity ``dst = a * src + b`` in the complex plane.

    Returns ``(scale, yaw_rad, translation_xy)``. The complex multiply by
    ``a = s * exp(i * yaw)`` folds rotation and scale into one unknown, so the
    solve is a single least-squares ratio plus a mean offset.
    """
    src = np.asarray(source_xy, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    if src.shape[0] != dst.shape[0]:
        raise ValueError(
            f"source/target length mismatch: {src.shape[0]} vs {dst.shape[0]}")
    if src.shape[0] < 2:
        raise ValueError(
            f"a planar similarity needs at least 2 correspondences, "
            f"got {src.shape[0]}")
    z = src[:, 0] + 1j * src[:, 1]
    w = dst[:, 0] + 1j * dst[:, 1]
    # The offset b is a free parameter, so both sums must be centred —
    # otherwise the solve is biased by the raw distance of the path from the
    # visual origin (which STEP 5 arbitrarily pins at the first keyframe).
    z_centred = z - z.mean()
    w_centred = w - w.mean()
    denominator = float(np.sum(np.abs(z_centred) ** 2))
    if denominator <= 1e-18:
        raise ValueError("degenerate source points: zero planar variance")
    a = np.sum(w_centred * np.conj(z_centred)) / denominator
    b = w.mean() - a * z.mean()
    return float(abs(a)), float(np.angle(a)), np.array([b.real, b.imag])


def estimate_planar_similarity(source: np.ndarray,
                               target: np.ndarray) -> SimilarityTransform:
    """Planar (yaw-only) similarity for a nadir-looking trajectory.

    The visual path is treated as flat: source x/y map onto global east/north
    through a 2D similarity (scale, yaw, offset) and source z maps onto global
    up with the same scale, with the height offset centred on the data.

    This is exact when the camera looks straight down — the common mapping
    flight — and is better conditioned than the full 3D fit, because a nearly
    planar path leaves the out-of-plane rotation weakly observable. It is wrong
    for a tilted or oblique camera, where the visual x/y axes are not
    horizontal; use mode ``3d`` there.
    """
    src, dst = _as_point_arrays(source, target)
    scale, yaw, t_xy = _similarity_2d(src[:, :2], dst[:, :2])
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos_yaw, -sin_yaw, 0.0],
                  [sin_yaw, cos_yaw, 0.0],
                  [0.0, 0.0, 1.0]])
    t = np.array([t_xy[0], t_xy[1],
                  float(dst[:, 2].mean() - scale * src[:, 2].mean())])
    return SimilarityTransform(scale, R, t)


def _resolve_estimator(mode: str):
    """Return the point-to-point fitter for an alignment mode."""
    key = (mode or "3d").strip().lower()
    if key == "3d":
        return estimate_similarity
    if key == "2d":
        return estimate_planar_similarity
    raise ValueError(
        f"unknown alignment mode {mode!r} — expected one of "
        f"{' | '.join(ALIGNMENT_MODES)}")


def _hypothesis(source: np.ndarray, target: np.ndarray, min_samples: int,
                estimator, rng: np.random.Generator) -> SimilarityTransform | None:
    """Fit one random minimal subset; ``None`` when that subset is degenerate.

    A minimal sample of random correspondences is routinely degenerate (three
    collinear camera centres, two identical GPS fixes), so a failed sample is
    an expected outcome, not an error.
    """
    n = source.shape[0]
    indices = (rng.choice(n, size=min_samples, replace=False)
               if n > min_samples else np.arange(n))
    try:
        return estimator(source[indices], target[indices])
    except (ValueError, np.linalg.LinAlgError):
        return None


def inlier_mask(transform: SimilarityTransform, source: np.ndarray,
                target: np.ndarray, threshold_m: float) -> np.ndarray:
    """Boolean mask of correspondences within ``threshold_m`` of the fit."""
    if threshold_m <= 0.0:
        raise ValueError(
            f"inlier threshold must be positive, got {threshold_m}")
    return similarity_residuals(transform, source, target) <= threshold_m


def refine_similarity(transform: SimilarityTransform, source: np.ndarray,
                      target: np.ndarray, mask: np.ndarray,
                      threshold_m: float, estimator=None) -> tuple[SimilarityTransform, np.ndarray]:
    """Re-fit the similarity on the inliers and re-evaluate the mask.

    One closed-form re-fit is enough: RANSAC picks a hypothesis that is
    correct but noisy, because it is fitted to three points. Re-fitting on the
    full inlier set uses every accepted correspondence and drops the sample
    noise; the mask is recomputed afterwards so the reported inlier set always
    matches the reported transform. ``estimator`` selects the alignment mode
    (3D by default); returns the inputs unchanged when the inliers are too few
    to re-fit or the subset is degenerate.
    """
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if int(selected.sum()) < MIN_SIMILARITY_SAMPLES:
        return transform, selected
    fit = estimate_similarity if estimator is None else estimator
    try:
        refined = fit(source[selected], target[selected])
    except (ValueError, np.linalg.LinAlgError):
        return transform, selected
    return refined, inlier_mask(refined, source, target, threshold_m)


def ransac_similarity(source: np.ndarray, target: np.ndarray,
                      inlier_threshold_m: float = 2.0,
                      iterations: int = 1000,
                      min_samples: int = MIN_SIMILARITY_SAMPLES,
                      mode: str = "3d",
                      seed: int | None = 42) -> AlignmentFit:
    """Robust visual->global similarity: RANSAC, then a closed-form re-fit.

    Hypotheses are scored by inlier count, ties broken by the lower median
    residual, and the winner is refined on its inliers. Outliers here are not
    hypothetical: consumer GPS jumps, fix dropout and pose/timestamp jitter
    routinely put a few correspondences metres off the track, and a plain
    least-squares fit would smear the whole reconstruction with them.

    The returned fit is marked ``success=False`` (with a ``reject_reason``) when
    there are too few correspondences, when no sampled hypothesis could be
    fitted, or when too few inliers survive — in that case the transform is
    the identity and must not be used to georeference anything.
    """
    estimator = _resolve_estimator(mode)
    src, dst = _as_point_arrays(source, target)
    n = src.shape[0]
    if n < min_samples:
        return AlignmentFit(
            transform=identity_transform(), n_correspondences=n,
            inlier_mask=np.zeros(n, dtype=bool), n_inliers=0, rmse_m=0.0,
            median_residual_m=0.0, max_residual_m=0.0, success=False,
            reject_reason="too_few_correspondences")
    if iterations < 1:
        raise ValueError(f"ransac iterations must be >= 1, got {iterations}")
    if min_samples < 2:
        raise ValueError(f"min_samples must be >= 2, got {min_samples}")

    rng = np.random.default_rng(seed)
    best: AlignmentFit | None = None
    used = 0
    for used in range(1, iterations + 1):
        candidate = _hypothesis(src, dst, min_samples, estimator, rng)
        if candidate is None:
            continue
        mask = inlier_mask(candidate, src, dst, inlier_threshold_m)
        n_inliers = int(mask.sum())
        rmse, median, worst = residual_stats(
            similarity_residuals(candidate, src, dst)[mask])
        if best is None or (n_inliers, -median) > (best.n_inliers,
                                                   -best.median_residual_m):
            best = AlignmentFit(
                transform=candidate, n_correspondences=n, inlier_mask=mask,
                n_inliers=n_inliers, rmse_m=rmse, median_residual_m=median,
                max_residual_m=worst, iterations=used)
    if best is None:
        return AlignmentFit(
            transform=identity_transform(), n_correspondences=n,
            inlier_mask=np.zeros(n, dtype=bool), n_inliers=0, rmse_m=0.0,
            median_residual_m=0.0, max_residual_m=0.0, iterations=used,
            success=False, reject_reason="no_valid_hypothesis")

    refined, mask = refine_similarity(best.transform, src, dst,
                                      best.inlier_mask, inlier_threshold_m,
                                      estimator)
    residuals = similarity_residuals(refined, src, dst)
    n_inliers = int(mask.sum())
    rmse, median, worst = residual_stats(residuals[mask])
    if (n_inliers, -rmse) >= (best.n_inliers, -best.rmse_m):
        best = AlignmentFit(
            transform=refined, n_correspondences=n, inlier_mask=mask,
            n_inliers=n_inliers, rmse_m=rmse, median_residual_m=median,
            max_residual_m=worst, iterations=used)
    if best.n_inliers < max(MIN_SIMILARITY_SAMPLES,
                            min_samples):
        best.success = False
        best.reject_reason = "insufficient_inliers"
    return best


@dataclass(frozen=True)
class PoseFixPair:
    """One accepted camera pose paired with the GPS fix nearest in time.

    ``dt_s`` is ``fix.timestamp_s - pose.timestamp_s`` (signed), kept so the
    report can show whether the pairing or the flight dynamics dominated the
    alignment residual.
    """

    pose: "CameraPose"
    fix: "MetricFix"
    dt_s: float


def associate_by_timestamp(poses: list, fixes: list,
                           max_gap_s: float = 0.5) -> list[PoseFixPair]:
    """Pair accepted camera poses with GPS fixes by nearest timestamp.

    The video and the flight log are independent recordings, so a pose has no
    GPS fix attached to it — only a time. Pairs further apart than
    ``max_gap_s`` are dropped rather than guessed at: a pose sampled at 2 fps
    and a log at 1 Hz can differ by half a second legitimately, but beyond
    that the association is fiction and would poison the fit.

    Candidates are consumed in order of increasing ``|dt|`` and one-to-one, so
    when two poses compete for the same fix the closest one wins. The result is
    returned in pose time order. Rejected poses and poses without a centre are
    skipped.
    """
    if max_gap_s < 0.0:
        raise ValueError(f"max_time_gap_s must be >= 0, got {max_gap_s}")
    if not fixes:
        return []
    fix_times = np.asarray([f.timestamp_s for f in fixes], dtype=np.float64)

    candidates: list[tuple[float, int, int, object]] = []
    for pose_index, pose in enumerate(poses):
        if not getattr(pose, "kept", False) or getattr(pose, "C", None) is None:
            continue
        fix_index = int(np.argmin(np.abs(fix_times - pose.timestamp_s)))
        dt = float(fix_times[fix_index] - pose.timestamp_s)
        if abs(dt) <= max_gap_s:
            candidates.append((abs(dt), pose_index, fix_index, dt))
    # Closest pairing first, so the winner of a contested fix is deterministic.
    candidates.sort(key=lambda c: (c[0], c[1]))

    used_fixes: set[int] = set()
    pairs: list[PoseFixPair] = []
    for _abs_dt, pose_index, fix_index, dt in candidates:
        if fix_index in used_fixes:
            continue
        used_fixes.add(fix_index)
        pairs.append(PoseFixPair(pose=poses[pose_index], fix=fixes[fix_index],
                                 dt_s=dt))
    pairs.sort(key=lambda p: p.pose.timestamp_s)
    return pairs
