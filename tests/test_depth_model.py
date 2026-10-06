"""STEP 9 — backend + inference tests: shapes, finite, confidence, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from src.depth.inference import depth_stats, ensure_finite, predict_single
from src.depth.model import (DummyDepthBackend, get_backend, resolve_device)


def _rgb(h=24, w=32, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def test_resolve_device_contract():
    assert resolve_device("auto") in ("cpu", "cuda")
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_dummy_backend_shapes_finite():
    backend = DummyDepthBackend(seed=42)
    depth = backend.predict(_rgb())
    assert depth.shape == (24, 32)
    assert bool(np.all(np.isfinite(depth)))
    assert depth.min() > 0


def test_dummy_backend_deterministic():
    img = _rgb()
    a = DummyDepthBackend(seed=7).predict(img)
    b = DummyDepthBackend(seed=7).predict(img)
    assert np.array_equal(a, b)


def test_get_backend_dummy_and_unknown():
    assert get_backend("dummy").name == "dummy"
    # auto without torch/transformers falls back to dummy with a warning
    assert get_backend("auto").name == "dummy"
    with pytest.raises(ValueError):
        get_backend("nope")


def test_ensure_finite_fills_nonfinite():
    d = np.ones((4, 4), dtype=np.float32)
    d[0, 0] = np.nan
    d[1, 1] = np.inf
    out = ensure_finite(d)
    assert bool(np.all(np.isfinite(out)))
    with pytest.raises(ValueError):
        ensure_finite(np.full((2, 2), np.nan, dtype=np.float32))


def test_predict_single_shapes_finite_confidence_range():
    img = _rgb(24, 32)
    depth, conf = predict_single(img, DummyDepthBackend(seed=42),
                                 input_size=16)
    assert depth.shape == (24, 32)  # resized back to source size
    assert conf.shape == (24, 32)
    assert bool(np.all(np.isfinite(depth)))
    assert bool(np.all(conf >= 0.0)) and bool(np.all(conf <= 1.0))


def test_predict_single_determinism_smoke():
    img = _rgb(16, 16, seed=3)
    backend = DummyDepthBackend(seed=123)
    d1, c1 = predict_single(img, backend, input_size=12)
    d2, c2 = predict_single(img, backend, input_size=12)
    assert np.array_equal(d1, d2) and np.array_equal(c1, c2)


def test_depth_stats_valid_flag():
    good = np.linspace(0.5, 1.5, 100, dtype=np.float32).reshape(10, 10)
    s = depth_stats(good)
    assert s["valid"] is True and s["finite_fraction"] == pytest.approx(1.0)
    bad = np.full((4, 4), np.nan, dtype=np.float32)
    s2 = depth_stats(bad)
    assert s2["valid"] is False and s2["finite_fraction"] == pytest.approx(0.0)
