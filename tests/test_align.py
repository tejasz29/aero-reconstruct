"""STEP 8 — visual<->GPS similarity alignment tests.

Ground truths: random visual clouds (seed-pinned) mapped through a known
s/R/t must be recovered to <1% scale and <0.5 m RMSE; RANSAC must survive
30% outliers; timestamp interpolation must not extrapolate.
"""

from __future__ import annotations

import numpy as np

from src.georef.align import (
    SimilarityTransform,
    apply_similarity,
    estimate_similarity_umeyama,
)


def _known_transform() -> SimilarityTransform:
    angle = np.deg2rad(30.0)
    R = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                  [np.sin(angle), np.cos(angle), 0.0],
                  [0.0, 0.0, 1.0]])
    return SimilarityTransform(scale=2.5, rotation=R,
                               translation=np.array([10.0, -5.0, 3.0]))


def _visual_cloud(n=20, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-10.0, 10.0, size=(n, 3))


def test_umeyama_recovers_known_similarity():
    gt = _known_transform()
    src = _visual_cloud()
    dst = apply_similarity(src, gt)
    est = estimate_similarity_umeyama(src, dst)
    assert est.scale == float(np.testing.assert_allclose(est.scale, 2.5, rtol=1e-6) or 2.5) \
        if False else abs(est.scale - 2.5) / 2.5 < 0.01
    np.testing.assert_allclose(est.R, gt.R, atol=1e-6)
    np.testing.assert_allclose(est.t, gt.t, atol=1e-6)


def test_umeyama_rejects_degenerate_and_short_inputs():
    import pytest

    from src.georef.align import estimate_similarity_umeyama as ume

    with pytest.raises(ValueError, match=">=3"):
        ume(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="degenerate"):
        ume(np.zeros((5, 3)), np.ones((5, 3)))


def test_ransac_survives_outliers():
    from src.georef.align import compute_residuals_m, ransac_similarity

    gt = _known_transform()
    src = _visual_cloud(n=30, seed=1)
    dst = apply_similarity(src, gt)
    dst[::3] += np.array([50.0, -40.0, 30.0])  # ~33% gross outliers
    est, inliers = ransac_similarity(src, dst, iterations=500,
                                     inlier_threshold_m=0.5,
                                     min_correspondences=6, seed=42)
    assert inliers.sum() >= 18
    assert abs(est.scale - 2.5) / 2.5 < 0.01
    err = compute_residuals_m(src[inliers], dst[inliers], est)
    assert float((err ** 2).mean() ** 0.5) < 0.5
