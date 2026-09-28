"""STEP 8 — residuals and RANSAC outlier rejection tests.

The realistic failure mode for this step is not a wrong transform on clean
data — it is a handful of corrupt correspondences (GPS spikes, fix dropout,
pose/timestamp jitter) dragging a least-squares fit across the whole
reconstruction. These tests inject exactly that, and pin the two guarantees
that matter: the true ``(s, R, t)`` survives the contamination, and the
rejected pairs are *named* rather than silently dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.georef.align import (
    AlignmentFit,
    SimilarityTransform,
    estimate_similarity,
    inlier_mask,
    ransac_similarity,
    refine_similarity,
    residual_stats,
    similarity_residuals,
)

TRUE_SCALE = 0.6


def make_case(n=30, seed=3, scale=TRUE_SCALE):
    """A clean visual/metric pair built from a random rotation + offset."""
    rng = np.random.default_rng(seed)
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    offset = np.array([150.0, -60.0, 25.0])
    a = np.linspace(0.0, 7.0, n)
    source = np.column_stack([np.cos(a), np.sin(a), 0.2 * a])
    target = scale * (source @ R.T) + offset
    return source, target, SimilarityTransform(scale, R, offset)


def test_residuals_are_zero_for_the_exact_transform():
    source, target, true = make_case()
    assert np.allclose(similarity_residuals(true, source, target), 0.0, atol=1e-9)
    # Shift one point by a known amount: exactly that distance shows up.
    bumped = target.copy()
    bumped[3] += np.array([0.3, 0.4, 0.0])
    residuals = similarity_residuals(true, source, bumped)
    assert residuals[3] == pytest.approx(0.5, rel=1e-9)
    assert np.allclose(np.delete(residuals, 3), 0.0, atol=1e-9)


def test_residual_stats_and_empty_input():
    rmse, median, worst = residual_stats(np.array([3.0, 4.0]))
    assert rmse == pytest.approx(np.sqrt(12.5))
    assert median == pytest.approx(3.5)
    assert worst == 4.0
    assert residual_stats(np.zeros(0)) == (0.0, 0.0, 0.0)


def test_inlier_mask_gates_on_the_residual_distance():
    source, target, true = make_case(n=10)
    noisy = target.copy()
    noisy[2] += np.array([5.0, 0.0, 0.0])
    mask = inlier_mask(true, source, noisy, threshold_m=1.0)
    assert mask.dtype == bool and mask.sum() == 9
    assert not mask[2]
    with pytest.raises(ValueError, match="must be positive"):
        inlier_mask(true, source, target, threshold_m=0.0)


def test_ransac_recovers_the_truth_despite_outliers():
    source, target, true = make_case()
    rng = np.random.default_rng(0)
    spikes = rng.choice(len(target), size=5, replace=False)
    contaminated = target.copy()
    contaminated[spikes] += rng.normal(scale=25.0, size=(len(spikes), 3))
    fit = ransac_similarity(source, contaminated, inlier_threshold_m=1.0,
                           iterations=500, seed=17)
    assert fit.success
    assert fit.reject_reason == ""
    assert fit.n_inliers == len(target) - len(spikes)
    assert set(fit.outlier_indices) == set(int(i) for i in spikes)
    assert fit.transform.scale == pytest.approx(TRUE_SCALE, rel=1e-3)
    assert np.allclose(fit.transform.R, true.R, atol=1e-3)
    assert fit.rmse_m < 0.1


def test_ransac_is_deterministic_for_a_fixed_seed():
    source, target, _ = make_case()
    rng = np.random.default_rng(4)
    contaminated = target.copy()
    contaminated[rng.choice(len(target), 4, replace=False)] += 40.0
    a = ransac_similarity(source, contaminated, inlier_threshold_m=1.0,
                          iterations=200, seed=99)
    b = ransac_similarity(source, contaminated, inlier_threshold_m=1.0,
                          iterations=200, seed=99)
    assert a.n_inliers == b.n_inliers
    assert np.allclose(a.transform.R, b.transform.R)
    assert np.allclose(a.transform.t, b.transform.t)


def test_refinement_beats_the_raw_hypothesis():
    source, target, true = make_case(n=25)
    noisy = target + np.random.default_rng(8).normal(scale=0.02,
                                                     size=target.shape)
    sample = np.array([0, 7, 19])
    rough = estimate_similarity(source[sample], noisy[sample])
    mask = inlier_mask(rough, source, noisy, threshold_m=1.0)
    refined, new_mask = refine_similarity(rough, source, noisy, mask, 1.0)
    before = float(np.max(similarity_residuals(rough, source, noisy)[mask]))
    after = float(np.max(similarity_residuals(refined, source, noisy)[new_mask]))
    assert after < before
    assert refined.scale == pytest.approx(true.scale, rel=0.01)


def test_refinement_keeps_inputs_when_inliers_are_too_few():
    source, target, true = make_case(n=6)
    mask = np.array([True, True] + [False] * 4)
    refined, new_mask = refine_similarity(true, source, target, mask, 1.0)
    assert refined is true
    assert new_mask.sum() == 2


def test_ransac_rejects_when_too_few_correspondences():
    source, target, _ = make_case(n=2)
    fit = ransac_similarity(source, target, min_samples=3)
    assert fit.success is False
    assert fit.reject_reason == "too_few_correspondences"
    assert fit.n_correspondences == 2
    assert fit.transform.scale == 1.0


def test_ransac_rejects_when_the_inliers_are_too_few():
    source, target, _ = make_case(n=20)
    far = target + np.arange(20)[:, None] * np.array([30.0, 0.0, 0.0])
    fit = ransac_similarity(source, far, inlier_threshold_m=1.0,
                            iterations=100, min_samples=3)
    assert fit.success is False
    assert fit.reject_reason in ("no_valid_hypothesis", "insufficient_inliers")
    # A rejected fit hands back the identity, never a plausible-looking
    # transform that a careless caller might georeference with.
    assert fit.transform.scale == 1.0
    assert np.allclose(fit.transform.R, np.eye(3))


def test_ransac_validates_its_own_arguments():
    source, target, _ = make_case(n=10)
    with pytest.raises(ValueError, match="iterations must be >= 1"):
        ransac_similarity(source, target, iterations=0)
    with pytest.raises(ValueError, match="min_samples must be >= 2"):
        ransac_similarity(source, target, min_samples=1)
    with pytest.raises(ValueError, match="length mismatch"):
        ransac_similarity(source, target[:3])


def test_ransac_planar_mode_recovers_a_flat_path():
    n = 12
    source = np.column_stack([np.linspace(0, 4, n), np.sin(np.linspace(0, 3, n)),
                              np.zeros(n)])
    yaw = np.deg2rad(-25.0)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                  [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    target = 2.0 * (source @ R.T) + np.array([10.0, 20.0, 30.0])
    fit = ransac_similarity(source, target, inlier_threshold_m=0.5,
                            iterations=200, mode="2d", seed=5)
    assert fit.success
    assert fit.transform.scale == pytest.approx(2.0, rel=1e-6)
    assert fit.n_inliers == n


def test_ransac_rejects_an_unknown_mode():
    source, target, _ = make_case(n=6)
    with pytest.raises(ValueError, match="unknown alignment mode"):
        ransac_similarity(source, target, mode="4d")


def test_alignment_fit_reports_a_stable_rejected_shape():
    fit = AlignmentFit(transform=SimilarityTransform(1.0, np.eye(3), np.zeros(3)),
                       n_correspondences=0, inlier_mask=np.zeros(0, dtype=bool),
                       n_inliers=0, rmse_m=0.0, median_residual_m=0.0,
                       max_residual_m=0.0, success=False,
                       reject_reason="too_few_correspondences")
    assert fit.inlier_ratio == 0.0
    assert fit.outlier_indices == []
