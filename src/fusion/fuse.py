"""STEP 11 — point-cloud fusion (confidence weighting, voxel downsample).

Merges STEP 10 per-frame clouds into one deduplicated scene. Pure-numpy
baseline; open3d is an optional fast path only. Metric-via-GPS-scale is
preserved; absolute CRS stays deferred to STEP 16.
"""

from __future__ import annotations

from src.common.logging_utils import get_logger

log = get_logger("sp3d.fusion")
