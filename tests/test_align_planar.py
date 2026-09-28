"""STEP 8 — planar (nadir) similarity tests.

A nadir-looking mapping flight has a flat trajectory, and for a flat path the
full 3D fit leaves the out-of-plane rotation barely observable. Mode ``2d``
solves the planar problem instead: a 2D similarity in ``(x, y)`` (scale, yaw,
offset) with the height offset centred on the data. These tests pin the
recovered scale and yaw against ground truth, and pin the boundary where the
planar model stops being valid.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.georef.align import (
    estimate_planar_similarity,
    similarity_residuals,
)

TRUE_SCALE = 1.75
TRUE_YAW = np.deg2rad(37.0)
TRUE_OFFSET = np.array([820.5, -14.25])


def make_flat_pair(n=10, scale=TRUE_SCALE, yaw=TRUE_YAW, offset=TRUE_OFFSET,
                   height=0.0):
    """A flat visual path and its planar image in the metric frame."""
    source = np.column_stack([np.linspace(0.0, 5.0, n),
                              np.linspace(0.0, 2.0, n) + 0.3 * np.sin(np.arange(n)),
                              np.full(n, height)])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
    target = scale * (source @ R.T) + np.array([offset[0], offset[1], 12.0])
    return source, target


def test_recovers_planar_scale_and_yaw():
    source, target = make_flat_pair()
    fit = estimate_planar_similarity(source, target)
    assert fit.scale == pytest.approx(TRUE_SCALE, rel=1e-9)
    yaw = float(np.arctan2(fit.R[1, 0], fit.R[0, 0]))
    assert yaw == pytest.approx(TRUE_YAW, rel=1e-6)
    assert np.allclose(fit.R, np.array([[np.cos(TRUE_YAW), -np.sin(TRUE_YAW), 0.0],
                                         [np.sin(TRUE_YAW), np.cos(TRUE_YAW), 0.0],
                                         [0.0, 0.0, 1.0]]), atol=1e-9)
    assert fit.t[0] == pytest.approx(TRUE_OFFSET[0], abs=1e-6)
    assert fit.t[1] == pytest.approx(TRUE_OFFSET[1], abs=1e-6)
    assert np.allclose(similarity_residuals(fit, source, target), 0.0, atol=1e-9)


def test_height_offset_is_centred_on_the_data():
    # A monocular path's z origin is arbitrary, so the height offset is a
    # mean, not a fixed constant: 30 m of "up" in the visual frame must not
    # change the in-plane solution.
    source, target = make_flat_pair(height=30.0)
    fit = estimate_planar_similarity(source, target)
    assert np.allclose(similarity_residuals(fit, source, target), 0.0, atol=1e-9)
    assert fit.scale == pytest.approx(TRUE_SCALE, rel=1e-9)


def test_planar_fit_tolerates_in_plane_noise():
    source, target = make_flat_pair(n=40)
    rng = np.random.default_rng(2)
    noisy = target.copy()
    noisy[:, :2] += rng.normal(scale=0.05, size=(noisy.shape[0], 2))
    fit = estimate_planar_similarity(source, noisy)
    assert fit.scale == pytest.approx(TRUE_SCALE, rel=5e-3)
    yaw = float(np.arctan2(fit.R[1, 0], fit.R[0, 0]))
    assert yaw == pytest.approx(TRUE_YAW, abs=np.deg2rad(0.5))


def test_planar_fit_needs_two_non_coincident_points():
    with pytest.raises(ValueError, match="at least 2 correspondences"):
        estimate_planar_similarity(np.zeros((1, 3)), np.zeros((1, 3)))
    same = np.zeros((5, 3))
    with pytest.raises(ValueError, match="zero planar variance"):
        estimate_planar_similarity(same, same + 1.0)


def test_planar_fit_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        estimate_planar_similarity(np.zeros((5, 3)), np.zeros((4, 3)))


def test_planar_model_cannot_explain_a_tilted_path():
    # The documented limit of mode 2d: with a roll in the true transform the
    # planar solution still returns *something*, and the residual is what
    # reveals that it is wrong. Silently accepting it would be the failure.
    source, _ = make_flat_pair()
    tilt = np.deg2rad(25.0)
    R = np.array([[np.cos(tilt), 0.0, np.sin(tilt)],
                  [0.0, 1.0, 0.0],
                  [-np.sin(tilt), 0.0, np.cos(tilt)]])
    tilted = TRUE_SCALE * (source @ R.T) + np.array([0.0, 0.0, 5.0])
    fit = estimate_planar_similarity(source, tilted)
    assert float(np.max(similarity_residuals(fit, source, tilted))) > 0.5
