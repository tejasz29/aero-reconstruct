"""STEP 9 — depth backends (Depth Anything family + CPU fallback).

The HuggingFace model is optional: when torch/transformers or weights are
missing the runner falls back to a deterministic dummy backend with a
warning, so the pipeline stays testable on CPU-only machines.
"""

from __future__ import annotations

from src.common.logging_utils import get_logger

log = get_logger("sp3d.depth")


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``auto | cuda | cpu`` to a concrete torch device string."""
    name = (requested or "auto").strip().lower()
    if name == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if name in ("cuda", "gpu"):
        return "cuda"
    if name == "cpu":
        return "cpu"
    raise ValueError(f"unknown device: {requested!r} (auto|cuda|cpu)")


class DepthBackend:
    """Minimal depth interface: HxWx3 uint8 -> HxW float32 relative depth."""

    name: str = "base"

    def predict(self, image) -> object:  # pragma: no cover - interface only
        raise NotImplementedError


class DummyDepthBackend(DepthBackend):
    """Deterministic CPU fallback (no weights): vertical gradient + texture."""

    name = "dummy"

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)

    def predict(self, image) -> object:
        """Deterministic relative depth: gradient + luma texture (finite)."""
        import numpy as np

        img = np.asarray(image)
        h, w = img.shape[:2]
        rng = np.random.default_rng(self.seed + h * 131 + w * 17)
        ys = np.linspace(0.0, 1.0, h, dtype=np.float64).reshape(h, 1)
        grad = np.broadcast_to(ys, (h, w)).copy()
        luma = img.mean(axis=2).astype(np.float64) / 255.0 if img.size else 0.0
        noise = rng.standard_normal((h, w)) * 0.02
        depth = 0.6 * grad + 0.3 * luma + 0.1 * (noise - noise.min())
        depth = depth - depth.min() + 0.1  # strictly positive, finite
        return depth.astype(np.float32)
