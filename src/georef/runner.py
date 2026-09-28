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
from src.georef.align import (
    MIN_SIMILARITY_SAMPLES,
    AlignmentFit,
    AlignmentResult,
    RtkInfo,
    accuracy_summary,
    associate_by_timestamp,
    build_aligned_rows,
    correspondence_arrays,
    pairing_dt_stats,
    ransac_similarity,
    rejected_fit,
    rtk_fixed_mask,
    rtk_info_from_types,
    write_aligned_csv,
    write_alignment_report,
    write_transform_json,
)
from src.georef.gps import read_metric_table
from src.sfm.visualize import read_poses_csv

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


@dataclass(frozen=True)
class AlignmentPaths:
    """Input and output locations of one alignment run."""

    poses_csv: Path
    gps_csv: Path
    aligned_csv: Path
    transform_json: Path
    report_json: Path


def _absolute(path: str | Path) -> Path:
    """Resolve a config-style relative path against the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def resolve_paths(cfg: dict, poses_csv: str | Path | None = None,
                  gps_csv: str | Path | None = None,
                  output_dir: str | Path | None = None) -> AlignmentPaths:
    """Resolve the STEP 5 / STEP 7 inputs and the three alignment outputs.

    The aligned trajectory lands next to the metric GPS file (it is data);
    the transform and the report land under ``paths.reports`` with the rest of
    the audit artefacts. ``output_name`` (``alignment.output_name``) drives all
    three file names so several alignments can coexist in one project.
    """
    name = str(get(cfg, "alignment.output_name", "aligned_trajectory"))
    traj_dir = _absolute(get(cfg, "paths.trajectory", "outputs/trajectory"))
    georef_dir = Path(output_dir) if output_dir else _absolute(
        get(cfg, "paths.georef", "outputs/georef"))
    if not georef_dir.is_absolute():
        georef_dir = PROJECT_ROOT / georef_dir
    reports_dir = _absolute(get(cfg, "paths.reports", "outputs/reports"))
    return AlignmentPaths(
        poses_csv=Path(poses_csv) if poses_csv else traj_dir / "camera_poses.csv",
        gps_csv=Path(gps_csv) if gps_csv else georef_dir / "gps_metric.csv",
        aligned_csv=georef_dir / f"{name}.csv",
        transform_json=reports_dir / f"{name}_transform.json",
        report_json=reports_dir / f"{name}_report.json",
    )


@dataclass(frozen=True)
class Correspondences:
    """Pose/GPS pairs ready for the fit, plus what the GPS tier turned out to be."""

    pairs: list
    rtk: RtkInfo
    n_fixes: int
    n_poses: int

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)


def build_correspondences(poses: list, table: "MetricTable",
                          policy: AlignmentPolicy) -> Correspondences:
    """Pair the accepted poses with metric fixes, RTK-first when available.

    The RTK filter runs *before* the pairing, not after: a log that is 90 %
    RTK-fixed and 10 % float has no business contributing its float fixes to
    a centimetre-scale alignment. When no fix is corrected — the normal
    consumer-drone case — every fix is kept and the tier is reported as such,
    because dropping the whole log would be worse than aligning to metre-level
    data and saying so.
    """
    mask = rtk_fixed_mask(table.fix_types) if policy.use_rtk_if_available else None
    if mask is not None and bool(mask.any()):
        fixes = [fix for fix, keep in zip(table.fixes, mask) if keep]
        types = [t for t, keep in zip(table.fix_types, mask) if keep]
        log.info("RTK/PPK fixes detected: using %d of %d fixes", len(fixes),
                 len(table.fixes))
    else:
        fixes, types = list(table.fixes), list(table.fix_types)
    info = rtk_info_from_types(types)
    pairs = associate_by_timestamp(poses, fixes, max_gap_s=policy.max_time_gap_s)
    log.info("paired %d of %d poses with fixes (max |dt| <= %.2f s)", len(pairs),
             sum(1 for p in poses if getattr(p, "kept", False)),
             policy.max_time_gap_s)
    return Correspondences(pairs=pairs, rtk=info, n_fixes=len(fixes), n_poses=len(poses))


def fit_alignment(corr: Correspondences,
                  policy: AlignmentPolicy) -> "AlignmentFit":
    """Fit the visual->global similarity under ``policy``.

    Two gates, and they are different kinds of gate: the residual gate
    (tightened to RTK class when corrected fixes are present) decides which
    correspondences are believable, and ``min_inliers`` decides whether
    enough of them survived to say anything at all. A fit that clears the
    first with three points and four thousand outliers is not an alignment.
    """
    source, target = correspondence_arrays(corr.pairs)
    gate = policy.gate_m(corr.rtk)
    fit = ransac_similarity(
        source, target, inlier_threshold_m=gate, iterations=policy.iterations,
        min_samples=max(MIN_SIMILARITY_SAMPLES, 2 if policy.mode == "2d" else 3),
        mode=policy.mode)
    if fit.success and fit.n_inliers < policy.min_inliers:
        fit.success = False
        fit.reject_reason = "insufficient_inliers"
    log.info("alignment fit: %d/%d inliers at a %.2f m gate (mode=%s, %d "
             "iterations)%s", fit.n_inliers, fit.n_correspondences, gate,
             policy.mode, fit.iterations,
             "" if fit.success else f" — REJECTED ({fit.reject_reason})")
    return fit


def _write_rejected(paths: AlignmentPaths, policy: AlignmentPolicy, mode: str,
                    crs: str, zone: str | None, poses_csv: str, gps_csv: str,
                    rtk: RtkInfo, n_pairs: int, reason: str,
                    iterations: int = 0) -> AlignmentResult:
    """Record a failed alignment instead of aborting the pipeline.

    A run that cannot be aligned still has to be explainable: the report is
    written with ``success: false``, the reason, the transform JSON is
    stamped ``status: rejected``, and no aligned trajectory is produced —
    a CSV full of identity-transformed positions would look like data while
    meaning nothing.
    """
    fit = rejected_fit(n_pairs, reason)
    log.error("alignment REJECTED (%s): %d correspondences — nothing is "
              "georeferenced", reason, n_pairs)
    transform_json = write_transform_json(
        fit.transform, paths.transform_json,
        extra={"status": "rejected", "reject_reason": reason, "crs": crs,
               "n_correspondences": n_pairs})
    result = AlignmentResult(
        transform=fit.transform, crs=crs, zone=zone, mode=mode,
        n_correspondences=n_pairs, n_inliers=0, rmse_m=0.0,
        median_residual_m=0.0, max_residual_m=0.0, rtk=rtk,
        accuracy=accuracy_summary(fit, rtk, n_pairs), rows=[],
        success=False, reject_reason=reason, aligned_csv="",
        transform_json=transform_json, iterations=iterations,
        pairing={"n_pairs": n_pairs}, poses_csv=poses_csv, gps_csv=gps_csv,
        inlier_mask=fit.inlier_mask)
    result.report_json = write_alignment_report(
        result, paths.report_json, extra={"policy": policy.as_dict()})
    return result


def run_alignment(cfg: dict,
                  poses_csv: str | Path | None = None,
                  gps_csv: str | Path | None = None,
                  output_dir: str | Path | None = None,
                  mode: str | None = None,
                  inlier_threshold_m: float | None = None) -> AlignmentResult:
    """STEP 8 entry point: align the SfM trajectory to the metric GPS track.

    Reads the STEP 5 poses and the STEP 7 metric fixes, fits the similarity
    robustly, and writes the aligned trajectory, the transform and the audit
    report. Tunables come from ``alignment.*`` in the config; ``mode`` and
    ``inlier_threshold_m`` override it for a one-off run.

    A run that cannot be aligned returns ``success=False`` with a
    ``reject_reason`` (``too_few_correspondences``, ``no_valid_hypothesis``,
    ``insufficient_inliers``) instead of raising: the caller decides whether
    an ungeoreferenced reconstruction is still worth writing out, and the
    report always says which case it is.
    """
    paths = resolve_paths(cfg, poses_csv, gps_csv, output_dir)
    policy = resolve_policy(cfg, mode, inlier_threshold_m)
    poses = read_poses_csv(paths.poses_csv)
    table = read_metric_table(paths.gps_csv)
    log.info("aligning %d poses against %d %s fixes (%s)", len(poses),
             len(table.fixes), table.crs, paths.poses_csv.name)

    corr = build_correspondences(poses, table, policy)
    if corr.n_pairs < policy.min_correspondences:
        return _write_rejected(
            paths, policy, policy.mode, table.crs, table.zone,
            str(paths.poses_csv), str(paths.gps_csv), corr.rtk, corr.n_pairs,
            "too_few_correspondences")

    fit = fit_alignment(corr, policy)
    if not fit.success:
        return _write_rejected(
            paths, policy, policy.mode, table.crs, table.zone,
            str(paths.poses_csv), str(paths.gps_csv), corr.rtk, corr.n_pairs,
            fit.reject_reason or "insufficient_inliers", iterations=fit.iterations)

    rows = build_aligned_rows(poses, corr.pairs, fit, fit.transform, table.crs)
    aligned_csv = write_aligned_csv(rows, paths.aligned_csv)
    transform_json = write_transform_json(
        fit.transform, paths.transform_json,
        extra={"crs": table.crs, "zone": table.zone or "",
               "alignment_rmse_m": round(fit.rmse_m, 4),
               "n_inliers": fit.n_inliers,
               "n_correspondences": fit.n_correspondences})
    result = AlignmentResult(
        transform=fit.transform, crs=table.crs, zone=table.zone,
        mode=policy.mode, n_correspondences=corr.n_pairs,
        n_inliers=fit.n_inliers, rmse_m=fit.rmse_m,
        median_residual_m=fit.median_residual_m,
        max_residual_m=fit.max_residual_m, rtk=corr.rtk,
        accuracy=accuracy_summary(fit, corr.rtk, corr.n_pairs), rows=rows,
        success=True, reject_reason="", aligned_csv=aligned_csv,
        transform_json=transform_json, iterations=fit.iterations,
        pairing=pairing_dt_stats(corr.pairs), poses_csv=str(paths.poses_csv),
        gps_csv=str(paths.gps_csv), inlier_mask=fit.inlier_mask)
    result.report_json = write_alignment_report(
        result, paths.report_json, extra={"policy": policy.as_dict()})
    log.info("alignment RMSE %.3f m over %d/%d inliers, scale %.4f m/unit",
             fit.rmse_m, fit.n_inliers, corr.n_pairs, fit.transform.scale)
    return result
