"""STEP 10 — shared fusion types.

One frame's depth becomes a colored cloud; all frames merge into the scene.
Points are metric via the STEP 8 GPS scale but NOT yet in an absolute CRS —
STEP 16 assigns ENU/UTM + origin. Every struct carries that flag.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameCloud:
    """One keyframe's unprojected points (metric-via-scale, Nx3 + Nx3 + N)."""

    frame_id: int
    filename: str
    n_points: int
    ply_path: str = ""
    mean_confidence: float = 0.0


SCALE_NOTE = (
    "Points are metric via the STEP 8 GPS scale (s) applied to relative "
    "depth; they are NOT yet in an absolute CRS — STEP 16 assigns ENU/UTM."
)


@dataclass(frozen=True)
class CloudStats:
    """Count + bounds summary for one cloud (for the report)."""

    n_points: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def is_valid(self) -> bool:
        """A cloud is usable when it has points and finite spread."""
        return self.n_points > 0 and self.xmax >= self.xmin
