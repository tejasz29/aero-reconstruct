"""STEP 9 — runner: keyframes -> relative depth + confidence maps.

Inputs:  ``data/frames/keyframes.csv`` (STEP 3 survivors) + images.
Outputs: ``outputs/depth/depth_*.npy`` + ``confidence_*.npy`` (+ previews)
         ``outputs/depth/depth_index.csv``
         ``outputs/reports/depth_report.json``

Depth is NOT metric: the runner stamps every artefact with the
relative-depth note so STEP 10 (unprojection) must apply the STEP 8 scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.logging_utils import get_logger

log = get_logger("sp3d.depth")


@dataclass(frozen=True)
class DepthPolicy:
    """Every tunable depth inference obeys, resolved from config."""

    backend: str = "dummy"
    model: str = "depth-anything-v2"
    device: str = "auto"
    input_size: int = 518
    store_confidence: bool = True
    seed: int = 42

    def as_dict(self) -> dict:
        return {"backend": self.backend, "model": self.model,
                "device": self.device, "input_size": self.input_size,
                "store_confidence": self.store_confidence, "seed": self.seed}
