"""STEP 8 — runner: visual trajectory <-> metric GPS alignment.

Inputs:  ``outputs/trajectory/camera_poses.csv`` (STEP 5, arbitrary scale)
         ``outputs/georef/gps_metric.csv``        (STEP 7, metric metres)
Outputs: ``outputs/georef/<output_name>.csv``               aligned trajectory
         ``outputs/reports/<output_name>_transform.json``   the (s, R, t)
         ``outputs/reports/<output_name>_report.json``      audit report

The runner is the config-driven glue: it resolves paths, resolves the
alignment policy from ``configs/default.yaml``, pairs poses with fixes, fits
the similarity robustly and writes the three artefacts above. The maths lives
in :mod:`src.georef.align`; this module only decides *what* to run and *where*
to put it, so the policy is auditable from the config file alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.config_loader import get
from src.common.logging_utils import get_logger
from src.common.paths import PROJECT_ROOT

log = get_logger("sp3d.georef")


@dataclass(frozen=True)
class AlignmentPolicy:
    """Every tunable the alignment obeys, resolved from config.

    Held in one frozen object so the report, the log line and the fit call
    all quote the same numbers — and so a test can pin a policy without
    building a config dict.
    """

    mode: str = "3d"
    iterations: int = 1000
    inlier_threshold_m: float = 2.0
    min_correspondences: int = 6
    min_inliers: int = 4
    max_time_gap_s: float = 0.5
    rtk_inlier_threshold_m: float = 0.5
    use_rtk_if_available: bool = True

    def gate_m(self, rtk) -> float:
        """Residual gate for the given GPS tier.

        RTK/PPK fixes are centimetre-class, so a 2 m gate would wave through
        pairs that are tens of centimetres wrong and report a sloppy
        alignment as clean. With corrected fixes the gate tightens to
        ``rtk_inlier_threshold_m``; without them the consumer gate stands.
        """
        if self.use_rtk_if_available and getattr(rtk, "available", False):
            return min(self.inlier_threshold_m, self.rtk_inlier_threshold_m)
        return self.inlier_threshold_m

    def as_dict(self) -> dict:
        """JSON-ready copy for the report."""
        return {
            "mode": self.mode,
            "ransac_iterations": self.iterations,
            "inlier_threshold_m": self.inlier_threshold_m,
            "min_correspondences": self.min_correspondences,
            "min_inliers": self.min_inliers,
            "max_time_gap_s": self.max_time_gap_s,
            "rtk_inlier_threshold_m": self.rtk_inlier_threshold_m,
            "use_rtk_if_available": self.use_rtk_if_available,
        }


def resolve_policy(cfg: dict, mode: str | None = None,
                   inlier_threshold_m: float | None = None) -> AlignmentPolicy:
    """Read the ``alignment.*`` section of the config into a policy."""
    defaults = AlignmentPolicy()
    requested = (mode or str(get(cfg, "alignment.mode", defaults.mode))).strip().lower()
    return AlignmentPolicy(
        mode=requested,
        iterations=int(get(cfg, "alignment.ransac_iterations", defaults.iterations)),
        inlier_threshold_m=float(
            inlier_threshold_m if inlier_threshold_m is not None
            else get(cfg, "alignment.inlier_threshold_m", defaults.inlier_threshold_m)),
        min_correspondences=int(
            get(cfg, "alignment.min_correspondences", defaults.min_correspondences)),
        min_inliers=int(get(cfg, "alignment.min_inliers", defaults.min_inliers)),
        max_time_gap_s=float(
            get(cfg, "alignment.max_time_gap_s", defaults.max_time_gap_s)),
        rtk_inlier_threshold_m=float(
            get(cfg, "alignment.rtk_inlier_threshold_m",
                defaults.rtk_inlier_threshold_m)),
        use_rtk_if_available=bool(
            get(cfg, "alignment.use_rtk_if_available", defaults.use_rtk_if_available)),
    )
