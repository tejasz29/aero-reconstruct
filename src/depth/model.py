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
