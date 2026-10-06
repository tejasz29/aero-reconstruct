"""STEP 9 — learned monocular depth inference (depth map + confidence)."""

from src.depth.confidence import confidence_from_depth, gradient_magnitude
from src.depth.inference import depth_stats, ensure_finite, predict_single
from src.depth.io import (load_depth_npy, read_depth_index, save_confidence_npy,
                          save_depth_npy, save_preview_png, write_depth_index,
                          write_depth_report)
from src.depth.model import (DepthBackend, DummyDepthBackend, get_backend,
                             resolve_device)
from src.depth.preprocess import load_image, normalize_chw, resize_to_model
from src.depth.runner import (DepthPaths, DepthPolicy, resolve_paths,
                              resolve_policy, run_depth_prediction)
from src.depth.types import DepthPrediction, DepthStats, RELATIVE_DEPTH_NOTE

__all__ = ["DepthBackend", "DepthPaths", "DepthPolicy", "DepthPrediction",
           "DepthStats", "DummyDepthBackend", "RELATIVE_DEPTH_NOTE",
           "confidence_from_depth", "depth_stats", "ensure_finite",
           "get_backend", "gradient_magnitude", "load_depth_npy",
           "load_image", "normalize_chw", "predict_single",
           "read_depth_index", "resolve_device", "resolve_paths",
           "resolve_policy", "resize_to_model", "run_depth_prediction",
           "save_confidence_npy", "save_depth_npy", "save_preview_png",
           "write_depth_index", "write_depth_report"]
