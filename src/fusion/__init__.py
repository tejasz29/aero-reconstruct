"""STEP 8/10/11 — depth->3D unprojection, fusion, outlier filtering."""

from src.fusion.fuse import (estimate_normals, estimate_normals_pca,
                             fuse_clouds, fused_stats, load_frame_clouds,
                             orient_normals,
                             recover_per_point_confidence, voxel_downsample,
                             voxel_keys)
from src.fusion.io import (CLOUD_INDEX_HEADER, load_ply,
                           load_ply_with_normals, read_cloud_index, save_ply,
                           validate_cloud, write_cloud_index,
                           write_fusion_report, write_unproject_report)
from src.fusion.runner import (FusionPaths, FusionPolicy, UnprojectPaths,
                               UnprojectPolicy, resolve_fusion_paths,
                               resolve_fusion_policy, resolve_paths,
                               resolve_policy, run_fusion, run_unprojection)
from src.fusion.types import SCALE_NOTE, CloudStats, FrameCloud
from src.fusion.unproject import (apply_pose, apply_scale, cloud_stats,
                                  colored_cloud, pixel_grid, reproject_points,
                                  unproject_depth, valid_mask)

__all__ = ["CLOUD_INDEX_HEADER", "SCALE_NOTE", "CloudStats", "FrameCloud",
           "FusionPaths", "FusionPolicy", "UnprojectPaths", "UnprojectPolicy",
           "apply_pose", "apply_scale", "cloud_stats", "colored_cloud",
           "estimate_normals", "estimate_normals_pca", "fuse_clouds",
           "fused_stats", "load_frame_clouds", "load_ply",
           "load_ply_with_normals", "orient_normals", "pixel_grid",
           "recover_per_point_confidence", "reproject_points",
           "resolve_fusion_paths", "resolve_fusion_policy", "resolve_paths",
           "resolve_policy", "run_fusion", "run_unprojection", "save_ply",
           "unproject_depth", "valid_mask", "validate_cloud",
           "voxel_downsample", "voxel_keys", "write_cloud_index",
           "write_fusion_report", "write_unproject_report"]
