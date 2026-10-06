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


RELATIVE_DEPTH_NOTE = (
    "Monocular depth is relative (unknown scale/shift); "
    "constrain with SfM poses + GPS scale before treating as metric."
)


@dataclass(frozen=True)
class DepthStats:
    """Finite-value summary of one depth map (for the index CSV/report)."""

    dmin: float
    dmax: float
    dmean: float
    finite_fraction: float

    def is_valid(self) -> bool:
        """A map is usable when every pixel is finite and spread is positive."""
        return (
            self.finite_fraction >= 1.0
            and self.dmax > self.dmin
            and self.dmax > 0.0
        )
