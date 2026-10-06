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


class HFDepthBackend(DepthBackend):
    """Depth Anything V2 via HuggingFace transformers (lazy, optional)."""

    name = "hf"

    def __init__(self, model_id: str = "depth-anything-v2", device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is not None:
            return self._pipe
        try:
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError(
                "transformers/torch missing for HF depth backend "
                f"({exc}); use backend=dummy on CPU-only machines"
            ) from exc
        log.info("loading HF depth model %s on %s", self.model_id, self.device)
        self._pipe = pipeline("depth-estimation", model=self.model_id,
                              device=self.device)
        return self._pipe

    def predict(self, image) -> object:
        import numpy as np
        from PIL import Image

        pipe = self._ensure_loaded()
        if isinstance(image, np.ndarray):
            pil = Image.fromarray(image.astype("uint8"))
        else:
            pil = image
        out = pipe(pil)
        depth = np.asarray(out["predicted_depth"], dtype=np.float32)
        return depth


def get_backend(name: str = "dummy", model_id: str = "depth-anything-v2",
                device: str = "auto", seed: int = 42) -> DepthBackend:
    """Factory: ``dummy`` always works; ``hf``/``auto`` try HF then fall back."""
    key = (name or "dummy").strip().lower()
    if key == "dummy":
        return DummyDepthBackend(seed=seed)
    if key in ("hf", "depth-anything-v2", "auto"):
        try:
            backend = HFDepthBackend(model_id=model_id, device=device)
            backend._ensure_loaded()
            return backend
        except Exception as exc:
            log.warning("HF depth unavailable (%s) — using dummy backend", exc)
            return DummyDepthBackend(seed=seed)
    raise ValueError(f"unknown depth backend: {name!r} (dummy|hf|auto)")
