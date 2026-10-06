"""STEP 9 — shared depth types.

Monocular depth is *relative* (up to an unknown scale/shift). It becomes
metric only when constrained by the STEP 5 poses + STEP 8 scale in STEP 10.
Every struct below carries that warning so no downstream stage can mistake
a depth map for metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DepthPrediction:
    """One keyframe's depth output (relative units, NOT metres)."""

    frame_id: int
    filename: str
    width: int
    height: int
    depth_path: str = ""
    confidence_path: str = ""
    depth_min: float = 0.0
    depth_max: float = 0.0
    depth_mean: float = 0.0
    confidence_mean: float = 0.0
