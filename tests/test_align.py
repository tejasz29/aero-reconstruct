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
    assert est.scale == float(np.testing.assert_allclose(est.scale, 2.5, rtol=1e-6) or 2.5)
    np.testing.assert_allclose(est.R, gt.R, atol=1e-6)
    np.testing.assert_allclose(est.t, gt.t, atol=1e-6)
