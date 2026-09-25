"""STEP 5 — two-view geometry (classical SfM core).

Estimates camera motion from feature correspondences with OpenCV's RANSAC
essential-matrix recovery, validates every pair by triangulating inlier
correspondences (positive-depth cheirality + reprojection error) and by
measuring median parallax, then chains the relative motions into a single
world frame anchored at the first accepted keyframe.

OpenCV conventions used throughout:

* ``findEssentialMat``/``recoverPose`` work on *undistorted* pixel
  coordinates with the pinhole camera matrix ``K`` (distortion is removed
  from the raw feature points before estimation).
* A relative pose ``(R, t)`` maps camera-1 coordinates onto camera-2
  coordinates: ``X2 = R @ X1 + t``.
* A camera pose ``(R_wc, C)`` stores the world->camera rotation and the
  camera centre ``C`` in the world frame (world = first accepted keyframe).

Scale note: a monocular camera recovers motion up to ONE unknown global
scale (the essential matrix returns unit-length translation). STEP 8
resolves the absolute metric scale via visual<->GPS similarity alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.common.logging_utils import get_logger

log = get_logger("sp3d.sfm")


@dataclass
class RelativePose:
    """Motion that takes camera-1 coordinates onto camera-2 coordinates."""

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inlier_count: int = 0
    mean_reproj_error_px: float = float("nan")
    median_parallax_px: float = 0.0
    valid: bool = False
    reject_reason: str = ""


@dataclass
class CameraPose:
    """One keyframe's pose in the trajectory world frame (== first frame)."""

    frame_id: int
    source_index: int
    timestamp_s: float
    filename: str
    kept: bool
    R_wc: np.ndarray | None = None     # world -> camera rotation (3x3)
    C: np.ndarray | None = None        # camera centre in world (3,)
    inliers: int = 0
    mean_reproj_error_px: float | None = None
    median_parallax_px: float | None = None
    reject_reason: str = ""


def _undistort_points(pts: np.ndarray, K: np.ndarray, dist: tuple) -> np.ndarray:
    """Rectify raw (distorted) pixels to pinhole pixels for geometry code."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    dist = np.asarray(dist, dtype=np.float64)
    if pts.shape[0] == 0 or not np.any(dist):
        return pts
    rect = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, None, K)
    return rect.reshape(-1, 2)


def _triangulate(points1: np.ndarray, points2: np.ndarray,
                 K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Triangulate undisturbed pixel pairs into camera-1 frame 3D points.

    ``P1 = K[I|0]``, ``P2 = K[R|t]``; returns (N, 3) points in camera-1
    coordinates ('X1').
    """
    p1 = cv2.triangulatePoints(K @ np.hstack([np.eye(3), np.zeros((3, 1))]),
                               K @ np.hstack([R, t.reshape(3, 1)]),
                               points1.reshape(-1, 2).T, points2.reshape(-1, 2).T)
    p1 = p1[:3] / p1[3]
    return p1.T


def _validate_pair(r1i: np.ndarray, r2i: np.ndarray, K: np.ndarray,
                   R: np.ndarray, t: np.ndarray,
                   min_inliers: int) -> tuple | None:
    """Triangulate and score one (R, t) hypothesis.

    Returns ``(front_count, X1, i1, i2, mean_error, parallax)`` where inlier
    points triangulate in front of both cameras and reproject tightly, or
    ``None`` when fewer than ``min_inliers`` correspondences survive.
    """
    X1 = _triangulate(r1i, r2i, K, R, t)
    z2 = (R @ X1.T).T + t
    front = (X1[:, 2] > 1e-3) & (z2[:, 2] > 1e-3)
    if int(front.sum()) < min_inliers:
        return None
    X1, i1, i2 = X1[front], r1i[front], r2i[front]
    p1_re = K @ X1.T
    p1_rows = (p1_re[:2] / p1_re[2]).T
    X2 = X1 @ R.T + t
    p2_rows = (K @ X2.T)[:2] / (K @ X2.T)[2]
    errors = np.linalg.norm(p1_rows - i1, axis=1) \
        + np.linalg.norm(p2_rows.T - i2, axis=1)
    parallax = float(np.median(np.linalg.norm(i2 - i1, axis=1)))
    return int(len(X1)), X1, i1, i2, float(np.mean(errors)), parallax


def _refine_pose(R: np.ndarray, t: np.ndarray, X1: np.ndarray,
                 i1: np.ndarray, i2: np.ndarray, K: np.ndarray,
                 iterations: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Newton refine (R, t) on fixed triangulated points (5 dof).

    Free parameters are the 3 Rodrigues angles plus the direction of ``t``
    (azimuth/elevation) — the essential matrix fixes translation only up to
    scale, and refining the direction keeps chaining scale-consistent.
    Residuals = per-point reprojection error in both views; the Jacobian is
    numeric (central differences). Returns the refined unit-translation pose.
    """
    t0 = t / (np.linalg.norm(t) + 1e-12)
    theta0 = np.r_[cv2.Rodrigues(R)[0].ravel(),
                   np.arctan2(t0[1], t0[0]),
                   np.arccos(np.clip(t0[2], -1.0, 1.0))]
    p1_fixed = K @ X1.T
    p1_fixed = (p1_fixed[:2] / p1_fixed[2]).T

    def unpack(theta):
        rvec, az, el = theta[:3], theta[3], theta[4]
        R_c, _ = cv2.Rodrigues(rvec)
        t_c = np.array([np.sin(el) * np.cos(az),
                        np.sin(el) * np.sin(az), np.cos(el)])
        return R_c, t_c

    def residual(theta):
        R_c, t_c = unpack(theta)
        p2 = K @ (X1 @ R_c.T + t_c).T
        p2 = (p2[:2] / p2[2]).T
        return np.concatenate([(p1_fixed - i1).ravel(), (p2 - i2).ravel()])

    theta = theta0.copy()
    r0 = residual(theta)
    cost = float(r0 @ r0)
    lam = 1e-3
    for _ in range(iterations):
        jac = np.zeros((len(r0), 5))
        for j in range(5):
            step = max(1e-6, 1e-4 * abs(theta[j]))
            tp, tm = theta.copy(), theta.copy()
            tp[j] += step
            tm[j] -= step
            jac[:, j] = (residual(tp) - residual(tm)) / (2.0 * step)
        normal = jac.T @ jac + lam * np.eye(5)
        try:
            delta = np.linalg.solve(normal, -(jac.T @ r0))
        except np.linalg.LinAlgError:
            break
        candidate = theta + delta
        rc = residual(candidate)
        new_cost = float(rc @ rc)
        if new_cost < cost:
            theta, r0, cost = candidate, rc, new_cost
            lam = max(lam / 3.0, 1e-9)
        else:
            lam *= 10.0
    R_ref, t_ref = unpack(theta)
    return R_ref, t_ref


def estimate_relative_pose(
    points1: np.ndarray,
    points2: np.ndarray,
    K: np.ndarray,
    distortion: tuple[float, ...] = (0.0,) * 5,
    ransac_threshold_px: float = 4.0,
    ransac_prob: float = 0.999,
    ransac_max_iter: int = 2000,
    min_inliers: int = 20,
    max_reprojection_error_px: float = 2.0,
    min_parallax_px: float = 3.0,
    transl_prior: np.ndarray | None = None,
) -> RelativePose:
    """Robustly estimate relative pose from two view's point correspondences.

    Feature points enter as raw (distorted) pixels; distortion is removed
    before the algebraic estimation. Among the RANSAC inliers, the
    triangulated scene must lie in front of both cameras with low
    reprojection error and usable parallax for the pose to be ``valid``.

    Small baselines make the *sign* of translation ambiguous even when
    cheirality holds; ``transl_prior`` (the unit direction of the previous
    accepted baseline) is used to flip ``t`` toward a temporally consistent
    direction when it produces an equally valid triangulation. Never
    raises; failures are described by ``reject_reason``.
    """
    result = RelativePose()
    p1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)
    if len(p1) < min_inliers or len(p1) != len(p2):
        result.reject_reason = "low_inliers"
        return result
    r1 = _undistort_points(p1, K, distortion)
    r2 = _undistort_points(p2, K, distortion)

    E, mask = cv2.findEssentialMat(r1, r2, K,
                                   method=cv2.RANSAC, prob=ransac_prob,
                                   threshold=ransac_threshold_px,
                                   maxIters=ransac_max_iter)
    if E is None:
        result.reject_reason = "failure"
        return result
    if mask is None:
        mask = np.ones(len(r1), dtype=np.uint8)
    mask = mask.ravel().astype(bool)
    result.inlier_count = int(mask.sum())
    if result.inlier_count < min_inliers:
        result.reject_reason = "low_inliers"
        return result

    _ok, R, t, _cheirality = cv2.recoverPose(E, r1[mask], r2[mask], K)
    t = t.ravel()

    candidates = [(R, t)]
    if (transl_prior is not None
            and float(t @ np.asarray(transl_prior, dtype=np.float64)) < 0):
        candidates.append((R, -t))
    best_R, best_t, best_score = None, None, None
    for R_c, t_c in candidates:
        score = _validate_pair(r1[mask], r2[mask], K, R_c, t_c, min_inliers)
        if score is None:
            continue
        if best_score is None or score[0] > best_score[0]:
            best_R, best_t, best_score = R_c, t_c, score
        elif (score[0] == best_score[0] and transl_prior is not None
              and float(t_c @ transl_prior) > float(best_t @ transl_prior)):
            best_R, best_t, best_score = R_c, t_c, score
    if best_score is None:
        result.reject_reason = "degenerate"
        return result
    R, t, (inlier_count, X1, i1, i2, _mean_error, _parallax) = (
        best_R, best_t, best_score)

    if inlier_count >= 8:
        R_ref, t_ref = _refine_pose(R, t, X1, i1, i2, K)
        refined = _validate_pair(i1, i2, K, R_ref, t_ref, min_inliers)
        if refined is not None and refined[4] < _mean_error:
            R, t = R_ref, t_ref
            inlier_count, X1, i1, i2, mean_error, parallax = refined
        else:
            mean_error, parallax = _mean_error, _parallax
    else:
        mean_error, parallax = _mean_error, _parallax

    result.R, result.t = R, t
    result.inlier_count = inlier_count
    result.mean_reproj_error_px = mean_error
    result.median_parallax_px = parallax
    if mean_error > max_reprojection_error_px:
        result.reject_reason = "high_reprojection"
    elif parallax < min_parallax_px:
        result.reject_reason = "low_parallax"
    else:
        result.reject_reason = ""
        result.valid = True
    return result


def chain_pose(R_wc_prev: np.ndarray, C_prev: np.ndarray,
               rel: RelativePose) -> tuple[np.ndarray, np.ndarray]:
    """Compose a chained world pose from a relative motion.

    With world frames ``X_cam = R_wc (X_w - C)`` and the relative map
    ``X2 = R_rel @ X1 + t_rel`` (camera-1 -> camera-2 coordinates), the
    world->camera rotation chains as ``R_wc_i = R_rel @ R_wc_{i-1}`` and the
    camera centre as ``C_i = C_{i-1} - (R_rel @ R_wc_{i-1}).T @ t_rel``.
    """
    R_wc = rel.R @ np.asarray(R_wc_prev, dtype=np.float64)
    C = (np.asarray(C_prev, dtype=np.float64)
         - (rel.R @ np.asarray(R_wc_prev, dtype=np.float64)).T @ rel.t)
    return R_wc, C


def estimate_absolute_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    distortion: tuple[float, ...] = (0.0,) * 5,
    ransac: bool = True,
    ransac_threshold_px: float = 4.0,
    iterations: int = 1000,
    confidence: float = 0.999,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """PnP pose of the camera (R_wc, camera centre C) from 3D<->2D matches.

    Returns ``None`` when the pose cannot be recovered (e.g. fewer than 4
    correspondences or a RANSAC failure).
    """
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    dist = np.asarray(distortion, dtype=np.float64)
    if len(obj) < 4:
        return None
    if len(obj) != len(img):
        raise ValueError("object_points and image_points must have equal length")
    if ransac:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K, dist, iterationsCount=iterations,
            reprojectionError=ransac_threshold_px, confidence=confidence)
        inlier_count = int(len(inliers)) if inliers is not None else 0
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
        inlier_count = len(obj)
    if not ok:
        return None
    R_cw, _ = cv2.Rodrigues(rvec)
    C = (-R_cw.T @ tvec.reshape(3)).reshape(3)
    return R_cw, C, inlier_count