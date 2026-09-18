"""STEP 4 — calibration runner: orchestrates the whole calibration step.

Config ``calibration.source`` selects the mode:

* ``provided``     — read a user/vendor-supplied intrinsics YAML
  (``calibration.provided_file``), validate it, write the canonical
  ``calibration/camera.yaml``.
* ``checkerboard`` — detect corners in ``calibration/images``, run
  ``cv2.calibrateCamera``.
* ``charuco``      — same flow via ``cv2.aruco``.

Every run writes the canonical camera model plus:
``outputs/reports/calibration_report.json`` and annotated views under
``outputs/reports/calibration/``. The report records the overall RMS, the
per-view RMS, rejected views, and the validation threshold verdict.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from src.calibration.charuco import calibrate_charuco
from src.calibration.checkerboard import (
    CalibrationResult,
    calibrate_checkerboard,
    detect_checkerboard,
)
from src.calibration.intrinsics import (
    Intrinsics,
    load_intrinsics,
    save_intrinsics,
)
from src.common.config_loader import get
from src.common.logging_utils import get_logger

log = get_logger("sp3d.calibration")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_complete(cam: dict[str, Any]) -> bool:
    values = [cam.get("fx"), cam.get("fy"), cam.get("cx"), cam.get("cy"),
              cam.get("image_width"), cam.get("image_height")]
    return all(v is not None for v in values)


def resolve_calibration(
    cfg: dict[str, Any],
    images_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[CalibrationResult | None, Intrinsics]:
    """Run the configured calibration mode and write artefacts.

    Returns ``(source_result, intrinsics)``; ``source_result`` is None for
    the ``provided`` mode (no board-based stats exist).
    """
    root = Path(cfg.get("paths", {}).get("root", ".")) if "root" in cfg.get("paths", {}) else _project_root()
    source = str(get(cfg, "calibration.source", "provided")).lower()
    if source not in ("provided", "checkerboard", "charuco"):
        raise ValueError(f"unknown calibration.source: {source!r} "
                         f"(provided | checkerboard | charuco)")

    if source != "provided":
        raise NotImplementedError(
            f"calibration.source={source!r} arrives in a later commit")

    provided = Path(get(cfg, "calibration.provided_file",
                        "calibration/camera_provided.yaml"))
    if not provided.is_absolute():
        provided = root / provided
    if not provided.is_file():
        raise FileNotFoundError(
            f"calibration.source=provided but {provided} missing — provide "
            f"intrinsics there or switch calibration.source to checkerboard/charuco")
    raw: dict[str, Any] = {}
    import yaml
    with open(provided, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cam = raw.get("camera") or raw
    if not _is_complete(cam):
        raise ValueError(
            f"incomplete provided intrinsics in {provided} — need fx, fy, "
            f"cx, cy, image_width, image_height")
    intrinsics = Intrinsics.from_dict(raw).validate()
    intrinsics.source = "provided"
    out = _write_outputs(cfg, root, intrinsics, None, output_path)
    return None, out


def _project_root() -> Path:
    from src.common.paths import PROJECT_ROOT
    return PROJECT_ROOT


def _write_outputs(cfg: dict[str, Any], root: Path, intrinsics: Intrinsics,
                   result: CalibrationResult | None,
                   output_path: str | Path | None) -> Intrinsics:
    """Save the canonical camera.yaml + JSON report."""
    out_file = Path(output_path) if output_path else Path(
        get(cfg, "calibration.file", "calibration/camera.yaml"))
    if not out_file.is_absolute():
        out_file = root / out_file
    save_intrinsics(intrinsics, out_file)

    reports_dir = Path(get(cfg, "paths.reports", "outputs/reports"))
    if not reports_dir.is_absolute():
        reports_dir = root / reports_dir
    cal_dir = reports_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    if result is not None:
        max_err = float(get(cfg, "calibration.validation.max_reprojection_error_px", 1.0))
        report = {
            "method": result.method,
            "rms_px": result.rms_px,
            "threshold_px": max_err,
            "within_threshold": result.rms_px <= max_err,
            "used_views": len(result.used_images),
            "rejected_views": [[str(p), why] for p, why in result.rejected_images],
            "per_view_rms_px": [[str(p), err, n]
                                for p, err, n in result.per_view_rms_px],
            "intrinsics": intrinsics.to_dict(),
            "calibrated_on": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        report_path = reports_dir / "calibration_report.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        log.info("calibration report -> %s (rms %.3f px, verdict %s)",
                 report_path, result.rms_px,
                 "PASS" if report["within_threshold"] else "CHECK")
    else:
        report = {"method": "provided", "intrinsics": intrinsics.to_dict(),
                  "calibrated_on": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(reports_dir / "calibration_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    if result is not None and result.rms_px > float(
            get(cfg, "calibration.validation.max_reprojection_error_px", 1.0)):
        log.warning("calibration RMS %.3f px exceeds threshold — model saved but "
                    "flagged CHECK in the report", result.rms_px)
    return intrinsics