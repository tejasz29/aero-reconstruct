"""STEP 9 — preprocess + confidence tests (shapes, range, determinism)."""

from __future__ import annotations

import numpy as np

from src.depth.confidence import (confidence_from_depth, gradient_magnitude,
                                  uniform_confidence, validate_confidence)
from src.depth.preprocess import load_image, normalize_chw, resize_to_model


def _rgb(h=24, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def test_load_image_roundtrip(tmp_path):
    from PIL import Image

    p = tmp_path / "frame.jpg"
    Image.fromarray(_rgb()).save(p)
    img = load_image(p)
    assert img.shape == (24, 32, 3) and img.dtype == np.uint8


def test_load_image_missing_raises(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        load_image(tmp_path / "nope.jpg")


def test_resize_keeps_aspect_and_caps_upscale():
    img = _rgb(100, 200)
    small = resize_to_model(img, input_size=50)
    assert max(small.shape[:2]) == 50
    assert small.shape[0] < small.shape[1]  # portrait ratio kept (100x200)
    tiny = _rgb(10, 10)
    capped = resize_to_model(tiny, input_size=518)
    assert max(capped.shape[:2]) <= 20  # 2x upscale cap


def test_normalize_chw_range():
    chw = normalize_chw(_rgb())
    assert chw.shape == (3, 24, 32)
    assert chw.min() >= 0.0 and chw.max() <= 1.0


def test_confidence_range_and_shape():
    depth = np.linspace(0.1, 2.0, 24 * 32, dtype=np.float32).reshape(24, 32)
    conf = confidence_from_depth(depth)
    assert conf.shape == depth.shape
    assert conf.dtype == np.float32
    assert bool(np.all(conf >= 0.0)) and bool(np.all(conf <= 1.0))
    assert bool(np.all(np.isfinite(conf)))
    validate_confidence(conf, depth.shape)


def test_confidence_flat_is_one():
    flat = np.ones((8, 8), dtype=np.float32)
    conf = confidence_from_depth(flat)
    assert np.allclose(conf, 1.0)


def test_uniform_confidence_contract():
    u = uniform_confidence((4, 5), 0.7)
    assert u.shape == (4, 5) and bool((u == 0.7).all())
    validate_confidence(u, (4, 5))
    import pytest

    with pytest.raises(ValueError):
        uniform_confidence((4, 4), 1.5)
    with pytest.raises(ValueError):
        validate_confidence(np.ones((2, 2)) * 2.0, (2, 2))


def test_gradient_magnitude_flat_zero():
    g = gradient_magnitude(np.ones((6, 6), dtype=np.float32))
    assert np.allclose(g, 0.0)
