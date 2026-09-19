"""STEP 5 — classical SfM: feature matching, relative pose, sequential stitching.

Public API::

    from src.sfm import run_reconstruction, estimate_relative_pose, ...  # noqa

Outputs: ``outputs/trajectory/camera_poses.csv`` +
``outputs/reports/trajectory_report.json`` (see ``runner``).
"""

from src.sfm.features import extract_features, match_features
from src.sfm.pose import (
    CameraPose,
    RelativePose,
    chain_pose,
    estimate_absolute_pose,
    estimate_relative_pose,
)
from src.sfm.runner import TrajectoryResult, run_reconstruction

__all__ = [
    "extract_features",
    "match_features",
    "CameraPose",
    "RelativePose",
    "chain_pose",
    "estimate_absolute_pose",
    "estimate_relative_pose",
    "TrajectoryResult",
    "run_reconstruction",
]