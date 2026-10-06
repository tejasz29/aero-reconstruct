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
