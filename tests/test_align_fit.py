"""STEP 8 — closed-form 3D similarity (Umeyama) tests.

Ground truth: a known ``(s, R, t)`` is applied to a synthetic SfM trajectory
and the fitter has to give it back — that is the whole promise of STEP 8,
because the SfM scale it recovers *is* the metric scale of the
reconstruction.

The synthetic path is a helix rather than a straight line on purpose: a
trajectory that spans all three axes equally is what makes a 7-dof
similarity observable, and a test that recovers ``s`` from a degenerate path
would be testing luck. The degenerate cases get their own tests, where the
required behaviour is a refusal, not a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.georef.align import (
    MIN_SIMILARITY_SAMPLES,
    estimate_similarity,
    similarity_residuals,
)

#: A non-trivial ground-truth similarity: 0.42 m per SfM unit, an exact 30 deg
#: yaw, and a 300 m offset — nothing about it is recoverable by inspection.
TRUE_SCALE = 0.42
_COS30, _SIN30 = np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))
TRUE_R = np.array([[_COS30, -_SIN30, 0.0],
                   [_SIN30, _COS30, 0.0],
                   [0.0, 0.0, 1.0]])
TRUE_T = np.array([300.0, -120.5, 42.0])


def make_source(n=12, seed=5, noise=0.005):
    """A well-spread visual path: a helix with a little SfM jitter."""
    a = np.linspace(0.0, 6.0, n)
    rng = np.random.default_rng(seed)
    return (np.column_stack([np.cos(a), np.sin(a), 0.15 * a])
            + rng.normal(scale=noise, size=(n, 3)))


def make_pair(n=12, seed=5, scale=TRUE_SCALE, R=TRUE_R, t=TRUE_T):
    """A visual path and the same path mapped through a known similarity."""
    source = make_source(n, seed)
    return source, scale * (source @ R.T) + t


def test_source_path_spans_all_three_axes():
    singular = np.linalg.svd(make_source() - make_source().mean(axis=0),
                             compute_uv=False)
    assert singular[1] > 0.5 * singular[0]
    assert singular[2] > 0.1 * singular[0]


def test_recovers_known_scale_rotation_and_translation():
    source, target = make_pair()
    fit = estimate_similarity(source, target)
    assert fit.scale == pytest.approx(TRUE_SCALE, rel=1e-9)
    assert np.allclose(fit.R, TRUE_R, atol=1e-9)
    assert np.allclose(fit.t, TRUE_T, atol=1e-9)
    assert np.allclose(similarity_residuals(fit, source, target), 0.0, atol=1e-9)


def test_fit_survives_metre_level_noise():
    # 5 cm of noise on a ~2 m wide path: the scale must stay within a
    # fraction of a percent, and every residual must stay at noise level.
    source, target = make_pair(n=40, seed=6)
    rng = np.random.default_rng(7)
    noisy = target + rng.normal(scale=0.05, size=target.shape)
    fit = estimate_similarity(source, noisy)
    assert fit.scale == pytest.approx(TRUE_SCALE, rel=0.015)
    assert np.allclose(fit.R, TRUE_R, atol=0.05)
    assert float(np.max(similarity_residuals(fit, source, noisy))) < 0.2


def test_planar_path_still_fits_a_full_3d_similarity():
    # A nadir-looking flight is coplanar but not collinear: the normal case,
    # not a special one.
    n = 15
    source = np.column_stack([np.linspace(0, 10, n), np.sin(np.linspace(0, 3, n)),
                              np.zeros(n)])
    target = TRUE_SCALE * (source @ TRUE_R.T) + TRUE_T
    fit = estimate_similarity(source, target)
    assert np.allclose(similarity_residuals(fit, source, target), 0.0, atol=1e-9)
    assert fit.scale == pytest.approx(TRUE_SCALE, rel=1e-9)
    assert np.allclose(fit.R, TRUE_R, atol=1e-9)


def test_rigid_mode_pins_the_scale_to_one():
    source, target = make_pair()
    fit = estimate_similarity(source, target, with_scale=False)
    assert fit.scale == 1.0
    assert np.allclose(fit.R, TRUE_R, atol=1e-9)
    # The rotation still lands; the scale is simply refused, so the residuals
    # are large — which is why the monocular path always asks for the scale.
    assert float(np.max(similarity_residuals(fit, source, target))) > 0.1


def test_mirrored_data_never_produce_a_mirror_transform():
    source, target = make_pair()
    mirrored = target.copy()
    mirrored[:, 0] *= -1.0
    fit = estimate_similarity(source, mirrored)
    # A camera-frame change of basis cannot mirror, so the reflection is
    # folded away: the rotation stays proper and the residual stays orders of
    # magnitude above the data's own noise — a visible failure rather than a
    # plausible-looking mirrored model.
    assert np.linalg.det(fit.R) == pytest.approx(1.0, abs=1e-9)
    assert float(np.max(similarity_residuals(fit, source, mirrored))) > 0.2


def test_degenerate_sources_are_refused():
    with pytest.raises(ValueError, match="at least 3 correspondences"):
        estimate_similarity(np.zeros((2, 3)), np.zeros((2, 3)))
    coincident = np.zeros((6, 3))
    with pytest.raises(ValueError, match="coincident or collinear"):
        estimate_similarity(coincident, coincident + 1.0)
    # A straight-line flight constrains a one-parameter family at best; it
    # must not be allowed to invent a rotation.
    collinear = np.column_stack([np.arange(6.0), np.zeros(6), np.zeros(6)])
    with pytest.raises(ValueError, match="coincident or collinear"):
        estimate_similarity(collinear, collinear * 2.0)


def test_correspondence_validation_rejects_mismatched_input():
    with pytest.raises(ValueError, match="length mismatch"):
        estimate_similarity(np.zeros((5, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="must all be finite"):
        estimate_similarity(np.full((5, 3), np.nan), np.zeros((5, 3)))


def test_min_samples_constant_matches_the_documented_contract():
    # Three pairs is the minimum for a 7-dof similarity, and the runner relies
    # on that number when it samples RANSAC hypotheses.
    assert MIN_SIMILARITY_SAMPLES == 3
