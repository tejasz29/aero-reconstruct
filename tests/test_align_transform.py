"""STEP 8 — similarity transform contract and serialisation tests.

The transform is the hand-off between STEP 8 and everything downstream, so
these tests pin the *meaning* of the model, not just its shape: rotation is
applied first, the offset is in metres afterwards, the inverse really is the
inverse, and a rejected fit can never be mistaken for a fitted one.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.georef.align import (
    AlignmentFit,
    SimilarityTransform,
    apply_similarity,
    apply_similarity_to_rotation,
    identity_transform,
    invert_similarity,
    rejected_fit,
    transform_from_dict,
    transform_to_dict,
)


def test_identity_transform_is_a_no_op():
    t = identity_transform()
    assert t.scale == 1.0
    assert np.allclose(t.R, np.eye(3))
    assert np.allclose(t.t, np.zeros(3))
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]])
    assert np.allclose(apply_similarity(t, pts), pts)


def test_transform_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="rotation must be 3x3"):
        SimilarityTransform(1.0, np.eye(2), np.zeros(3))
    with pytest.raises(ValueError, match="translation must have 3"):
        SimilarityTransform(1.0, np.eye(3), np.zeros(2))
    with pytest.raises(ValueError, match="scale must be positive"):
        SimilarityTransform(0.0, np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="scale must be positive"):
        SimilarityTransform(-2.0, np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        SimilarityTransform(1.0, np.full((3, 3), np.nan), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        SimilarityTransform(1.0, np.eye(3), np.array([np.inf, 0.0, 0.0]))


def test_transform_is_frozen_and_normalises_inputs():
    t = SimilarityTransform(2.0, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            [[1.0], [2.0], [3.0]])
    assert isinstance(t.R, np.ndarray) and t.R.dtype == np.float64
    assert t.t.shape == (3,)
    with pytest.raises(Exception):
        t.scale = 3.0


def test_apply_similarity_scales_then_rotates_then_translates():
    # 90 deg yaw: (x, y, z) -> (-y, x, z)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = SimilarityTransform(2.0, R, np.array([10.0, -5.0, 1.0]))
    out = apply_similarity(t, np.array([[1.0, 0.0, 0.0]]))
    assert np.allclose(out, [[10.0, -3.0, 1.0]])
    assert np.allclose(apply_similarity(t, np.zeros((0, 3))).shape, (0, 3))


def test_apply_similarity_to_rotation_drops_the_scale():
    R = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    t = SimilarityTransform(7.5, R, np.zeros(3))
    R_wc = np.eye(3)
    assert np.allclose(apply_similarity_to_rotation(t, R_wc), R.T)
    assert np.allclose(apply_similarity_to_rotation(t, R_wc).T
                       @ apply_similarity_to_rotation(t, R_wc), np.eye(3))


def test_invert_similarity_round_trips():
    rng = np.random.default_rng(11)
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    t = SimilarityTransform(0.37, R, np.array([120.0, -33.0, 8.0]))
    inv = invert_similarity(t)
    pts = rng.normal(size=(25, 3)) * 10.0
    assert np.allclose(apply_similarity(inv, apply_similarity(t, pts)), pts)
    assert inv.scale == pytest.approx(1.0 / 0.37)


def test_transform_dict_round_trip_keeps_the_model_explicit():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = SimilarityTransform(1.25, R, np.array([1.0, 2.0, 3.0]))
    payload = transform_to_dict(t)
    assert payload["model"] == ("X_global = scale * (R @ X_visual) "
                                "+ translation_m")
    assert payload["scale"] == 1.25
    assert payload["translation_m"] == [1.0, 2.0, 3.0]
    back = transform_from_dict(payload)
    assert back.scale == pytest.approx(1.25)
    assert np.allclose(back.R, R) and np.allclose(back.t, t.t)


def test_transform_from_dict_rejects_broken_payloads():
    with pytest.raises(ValueError, match="must be a dict"):
        transform_from_dict([1, 2, 3])
    with pytest.raises(ValueError, match="missing 'scale'"):
        transform_from_dict({"rotation": np.eye(3).tolist(),
                             "translation_m": [0, 0, 0]})
    with pytest.raises(ValueError, match="rotation must be 3x3"):
        transform_from_dict({"scale": 1.0, "rotation": np.eye(2).tolist(),
                             "translation_m": [0, 0, 0]})
    with pytest.raises(ValueError, match="translation must have 3"):
        transform_from_dict({"scale": 1.0, "rotation": np.eye(3).tolist(),
                             "translation_m": [0, 0]})


def test_rejected_fit_is_identity_and_flagged():
    fit = rejected_fit(4, "too_few_correspondences")
    assert fit.success is False
    assert fit.reject_reason == "too_few_correspondences"
    assert fit.n_correspondences == 4
    assert fit.n_inliers == 0
    assert not fit.inlier_mask.any()
    assert np.allclose(fit.transform.R, np.eye(3))
    assert fit.transform.scale == 1.0


def test_fit_inlier_ratio_and_outlier_indices():
    mask = np.array([True, True, False, True, False])
    fit = AlignmentFit(transform=identity_transform(), n_correspondences=5,
                       inlier_mask=mask, n_inliers=3, rmse_m=1.0,
                       median_residual_m=0.5, max_residual_m=4.0)
    assert fit.inlier_ratio == pytest.approx(0.6)
    assert fit.outlier_indices == [2, 4]
