"""STEP 5/6 — classical SfM: feature matching, relative pose, sequential
stitching, and trajectory visualisation.

Public API::

    from src.sfm import run_reconstruction, estimate_relative_pose, ...  # noqa

Outputs: ``outputs/trajectory/camera_poses.csv`` +
``outputs/reports/trajectory_report.json`` (see ``runner``); trajectory
figures in ``outputs/reports/`` (see ``visualize``).
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
from src.sfm.visualize import (
    TrajectoryPlots,
    camera_axes,
    plot_trajectory,
    read_poses_csv,
)

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
    "TrajectoryPlots",
    "camera_axes",
    "plot_trajectory",
    "read_poses_csv",
]