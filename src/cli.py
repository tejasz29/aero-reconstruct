"""Command-line entry point for single-pass-3d.

Usage (from the project root)::

    python -m src.cli version
    python -m src.cli doctor
    python -m src.cli init
    python -m src.cli extract-frames --video data/raw_videos/flight.mp4

or, after ``pip install -e .``::

    sp3d version | doctor | init | extract-frames ...

* ``version`` — print the package version.
* ``doctor``  — verify interpreter, folder layout, FFmpeg and optional
  heavy dependencies; exits non-zero if the core layout is broken.
* ``init``    — create any missing project directories (idempotent).
* ``extract-frames`` — STEP 2: probe a drone video and dump thinned
  frames + timestamps.csv.
* ``select-keyframes`` — STEP 3: score frames, drop bad/duplicate ones,
  write frame_scores.csv + keyframes.csv.
* ``calibrate`` — STEP 4: camera calibration (provided | checkerboard |
  charuco) -> validated calibration/camera.yaml + report.
* ``reconstruct-poses`` — STEP 5: track keyframes -> camera_poses.csv +
  trajectory report (SIFT/ORB, essential-matrix RANSAC, chained poses).
* ``show-trajectory`` — STEP 6: plot camera_poses.csv -> 3D + top-down
  figures under outputs/reports/ (sanity-check before heavy steps).
* ``convert-gps`` — STEP 7: project the GPS log into metric coordinates
  (ENU or UTM) -> outputs/georef/gps_metric.csv + report.

Pipeline-stage subcommands for later STEPS are added as those steps land.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys

from src.common.config_loader import get, load_config
from src.common.logging_utils import get_logger, setup_logging
from src.common.paths import PROJECT_ROOT, project_paths
from src.calibration.runner import resolve_calibration
from src.georef.gps import run_gps_conversion
from src.sfm.runner import run_reconstruction
from src.sfm.visualize import plot_trajectory, read_poses_csv
from src.video.frame_extractor import extract_frames
from src.video.keyframes import run_selection

VERSION = "0.1.0"

# Heavy third-party packages per pipeline area. Missing ones are WARNINGS
# at this stage — stages that need them fail loudly when they run.
OPTIONAL_DEPS = {
    "video I/O / features (opencv)": ["cv2", "numpy", "scipy"],
    "AI depth / segmentation (torch)": ["torch", "torchvision", "transformers", "ultralytics"],
    "3D (open3d)": ["open3d"],
    "geospatial (pyproj)": ["pyproj", "rasterio", "shapely", "laspy"],
    "data / plots (pandas)": ["pandas", "matplotlib", "tqdm"],
    "backend (fastapi)": ["fastapi", "uvicorn"],
}

log = get_logger("sp3d.cli")


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"single-pass-3d {VERSION}")
    return 0


def cmd_init(_args: argparse.Namespace) -> int:
    paths = project_paths().ensure_dirs()
    print(f"project initialised at {paths.root}")
    return 0


def _check_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Environment + layout health check. Returns 0 when the core is OK."""
    ok = True
    print(f"project root : {PROJECT_ROOT}")
    print(f"python       : {sys.version.split()[0]} (target: 3.10/3.11)")
    if sys.version_info[:2] not in ((3, 10), (3, 11)):
        print("  WARNING: interpreter is outside the 3.10/3.11 target — "
              "heavy wheels (torch/open3d) may be unavailable.")

    missing = project_paths().missing_dirs()
    if missing:
        ok = False
        print("missing directories (run `sp3d init`):")
        for path in missing:
            print(f"  MISSING {path}")
    else:
        print("directories  : OK (all 16 layout dirs present)")

    try:
        cfg = load_config()
        print(f"config       : OK ({get(cfg, 'project.name')} "
              f"{get(cfg, 'project.version')}, "
              f"{len(cfg)} top-level sections)")
    except Exception as exc:  # noqa: BLE001 — doctor must never crash
        ok = False
        print(f"config       : FAILED ({exc})")

    ffmpeg = shutil.which("ffmpeg")
    print(f"ffmpeg       : {ffmpeg if ffmpeg else 'NOT FOUND on PATH (needed from STEP 2)'}")

    print("optional dependencies:")
    for area, modules in OPTIONAL_DEPS.items():
        states = [f"{m}={'OK' if _check_module(m) else 'MISSING'}" for m in modules]
        print(f"  {area:38s} {', '.join(states)}")

    print("doctor result:", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


def cmd_extract_frames(args: argparse.Namespace) -> int:
    """STEP 2 — dump thinned frames + timestamps.csv for one video."""
    from pathlib import Path

    cfg = load_config(args.config) if args.config else load_config()
    frames_dir = Path(args.frames_dir) if args.frames_dir else Path(
        get(cfg, "paths.frames", "data/frames"))
    if not frames_dir.is_absolute():
        frames_dir = PROJECT_ROOT / frames_dir

    def _opt(attr: str, key: str, fallback: object) -> object:
        value = getattr(args, attr, None)
        return value if value is not None else get(cfg, f"video.{key}", fallback)

    result = extract_frames(
        args.video,
        frames_dir,
        target_fps=float(_opt("target_fps", "target_fps", 2.0)),
        max_frames=int(_opt("max_frames", "max_frames", 2000)),
        max_width=int(get(cfg, "video.image_max_width", 1920)),
        output_format=str(get(cfg, "video.output_format", "jpg")),
        jpeg_quality=int(get(cfg, "video.jpeg_quality", 95)),
    )
    i = result.info
    print(f"video : {i.path} ({i.width}x{i.height} @ {i.fps:.2f} fps, "
          f"{i.frame_count} frames, {i.duration_s:.1f}s)")
    print(f"frames: {len(result.frames)} -> {result.out_dir} (+ {result.csv_path.name})")
    return 0


def cmd_select_keyframes(args: argparse.Namespace) -> int:
    """STEP 3 — score frames and select keyframes."""
    from pathlib import Path

    cfg = load_config(args.config) if args.config else load_config()
    frames_dir = Path(args.frames_dir) if args.frames_dir else Path(
        get(cfg, "paths.frames", "data/frames"))
    if not frames_dir.is_absolute():
        frames_dir = PROJECT_ROOT / frames_dir

    result = run_selection(
        frames_dir,
        blur_threshold=float(get(cfg, "video.quality.blur_threshold", 100.0)),
        exposure_min=float(get(cfg, "video.quality.min_exposure_mean", 15.0)),
        exposure_max=float(get(cfg, "video.quality.max_exposure_mean", 240.0)),
        min_features=int(get(cfg, "video.quality.min_features", 300)),
        min_time_gap_s=float(get(cfg, "video.keyframes.min_time_gap_s", 0.0)),
        dedup_hamming_threshold=int(
            get(cfg, "video.keyframes.dedup_hamming_threshold", 5)),
        max_keep=int(get(cfg, "video.keyframes.max_keep", 600)),
        detector=str(get(cfg, "video.quality.detector", "orb")),
    )
    print(f"scored : {len(result.scored)} frames in {result.frames_dir}")
    print(f"kept   : {len(result.kept)} keyframes -> {result.keyframes_path.name}")
    print(f"scores : {result.scores_path.name} (with reject reasons)")
    if result.rejected_counts:
        print("rejected: " + ", ".join(
            f"{reason}={n}" for reason, n in sorted(result.rejected_counts.items())))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """STEP 4 — camera calibration (provided | checkerboard | charuco)."""
    from pathlib import Path

    cfg = load_config(args.config) if args.config else load_config()
    images_dir = None
    if args.images_dir:
        images_dir = Path(args.images_dir)
        if not images_dir.is_absolute():
            images_dir = PROJECT_ROOT / images_dir

    result, intrinsics = resolve_calibration(cfg, images_dir=images_dir,
                                             output_path=args.output)
    print(f"source : {intrinsics.source}")
    print(f"model  : fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} "
          f"cx={intrinsics.cx:.2f} cy={intrinsics.cy:.2f} "
          f"({intrinsics.width}x{intrinsics.height})")
    print(f"dist   : [{', '.join(f'{d:.5f}' for d in intrinsics.distortion)}]")
    if result is not None:
        threshold = float(get(cfg, "calibration.validation.max_reprojection_error_px", 1.0))
        verdict = "PASS" if result.rms_px <= threshold else "CHECK"
        print(f"views  : {len(result.used_images)} used, "
              f"{len(result.rejected_images)} rejected")
        print(f"rms    : {result.rms_px:.3f} px (threshold {threshold:.1f} -> {verdict})")
        if result.rejected_images:
            print("rejected:" + ", ".join(f" {p.name} ({why})"
                                          for p, why in result.rejected_images[:5]))
    print(f"output : {args.output or get(cfg, 'calibration.file', 'calibration/camera.yaml')}")
    report_dir = Path(get(cfg, "paths.reports", "outputs/reports"))
    print(f"report : {report_dir / 'calibration_report.json'} "
          f"(+ annotated views in {report_dir / 'calibration/'})")
    return 0


def cmd_reconstruct_poses(args: argparse.Namespace) -> int:
    """STEP 5 — track keyframes into camera_poses.csv + report."""
    from pathlib import Path

    cfg = load_config(args.config) if args.config else load_config()
    frames_dir = Path(args.frames_dir) if args.frames_dir else Path(
        get(cfg, "paths.frames", "data/frames"))
    if not frames_dir.is_absolute():
        frames_dir = PROJECT_ROOT / frames_dir
    keyframes_file = Path(args.keyframes_file) if args.keyframes_file else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    result = run_reconstruction(
        cfg, frames_dir=frames_dir, keyframes_file=keyframes_file,
        intrinsics=args.camera, output_dir=output_dir)
    mean = result.mean_reproj_error_px
    print(f"backend : {result.backend}")
    print(f"keyframes: {len(result.poses)} tracked "
          f"(accepted={len(result.kept)}, rejected={len(result.rejected)})")
    if result.rejected:
        print("rejected: " + ", ".join(
            f"{reason}={n}" for reason, n in sorted(result.rejected_counts.items())))
    print(f"mean_reproj_error_px: {mean:.3f}" if mean is not None
          else "mean_reproj_error_px: -")
    print(f"poses   : {result.poses_path}")
    print(f"report  : {result.report_path}")
    return 0


def cmd_show_trajectory(args: argparse.Namespace) -> int:
    """STEP 6 — plot the estimated camera path (camera_poses.csv -> PNGs)."""
    from pathlib import Path

    cfg = load_config(args.config) if args.config else load_config()
    poses_path = Path(args.poses_csv) if args.poses_csv else Path(
        get(cfg, "paths.trajectory", "outputs/trajectory")) / "camera_poses.csv"
    if not poses_path.is_absolute():
        poses_path = PROJECT_ROOT / poses_path
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        get(cfg, "paths.reports", "outputs/reports"))
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    plots = plot_trajectory(read_poses_csv(poses_path), out_dir, cfg,
                            topdown_projection=args.topdown)
    print(f"poses   : {poses_path}")
    for path in plots.paths:
        print(f"figure  : {path}")
    return 0


def cmd_convert_gps(args: argparse.Namespace) -> int:
    """STEP 7 — project the GPS log into metric coordinates + report."""
    from pathlib import Path

    cfg = load_config(args.config) if args.config else load_config()
    gps_file = Path(args.gps_file) if args.gps_file else Path(
        get(cfg, "gps.file", "data/gps/gps.csv"))
    if not gps_file.is_absolute():
        gps_file = PROJECT_ROOT / gps_file
    output_dir = Path(args.output_dir) if args.output_dir else None

    result = run_gps_conversion(
        cfg, gps_file=gps_file, output_dir=output_dir,
        crs=args.crs, zone_override=args.utm_zone)
    zone = result.zone if result.zone is not None else "-"
    print(f"source : {result.source}")
    print(f"fixes  : {result.n_fixes}")
    print(f"crs    : {result.crs} (zone {zone})")
    print(f"origin : {result.origin.latitude:.7f}, {result.origin.longitude:.7f}, "
          f"{result.origin.altitude_m:.2f} m")
    print(f"extent : {result.extent_m:.1f} m")
    print(f"output : {result.metric_csv}")
    print(f"report : {result.report_json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sp3d",
        description="Single-pass drone video -> 3D reconstruction (offline MVP).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print version and exit.")
    sub.add_parser("doctor", help="check environment and project layout.")
    sub.add_parser("init", help="create missing project directories.")

    ex = sub.add_parser("extract-frames", help="STEP 2: dump thinned frames + timestamps.csv.")
    ex.add_argument("--video", required=True, help="input drone video file.")
    ex.add_argument("--frames-dir", default=None,
                    help="output dir (default: paths.frames from config).")
    ex.add_argument("--config", default=None, help="run config YAML (default.yaml + merge).")
    ex.add_argument("--target-fps", type=float, default=None,
                    help="override video.target_fps.")
    ex.add_argument("--max-frames", type=int, default=None,
                    help="override video.max_frames.")

    sk = sub.add_parser("select-keyframes",
                        help="STEP 3: score frames and select keyframes.")
    sk.add_argument("--frames-dir", default=None,
                    help="frames dir with timestamps.csv (default: from config).")
    sk.add_argument("--config", default=None, help="run config YAML (default.yaml + merge).")

    cl = sub.add_parser("calibrate",
                        help="STEP 4: camera calibration (provided|checkerboard|charuco).")
    cl.add_argument("--images-dir", default=None,
                    help="calibration photos dir (default: calibration.images_dir).")
    cl.add_argument("--output", default=None,
                    help="output camera model path (default: calibration.file).")
    cl.add_argument("--config", default=None, help="run config YAML (default.yaml + merge).")

    rp = sub.add_parser("reconstruct-poses",
                        help="STEP 5: track keyframes and write camera_poses.csv.")
    rp.add_argument("--frames-dir", default=None,
                    help="frames dir with keyframes.csv (default: paths.frames).")
    rp.add_argument("--keyframes-file", default=None,
                    help="keyframes.csv to read (default: <frames-dir>/keyframes.csv).")
    rp.add_argument("--camera", default=None,
                    help="camera model YAML (default: calibration.file).")
    rp.add_argument("--output-dir", default=None,
                    help="output dir (default: paths.trajectory).")
    rp.add_argument("--config", default=None, help="run config YAML (default.yaml + merge).")

    vt = sub.add_parser("show-trajectory",
                        help="STEP 6: plot camera poses into PNGs (3D + top-down).")
    vt.add_argument("--poses-csv", default=None,
                    help="pose CSV (default: <paths.trajectory>/camera_poses.csv).")
    vt.add_argument("--output-dir", default=None,
                    help="figures output dir (default: paths.reports).")
    vt.add_argument("--topdown", default=None, choices=["xy", "xz", "yz"],
                    help="override visualization.topdown_projection.")
    vt.add_argument("--config", default=None, help="run config YAML (default.yaml + merge).")

    cg = sub.add_parser("convert-gps",
                        help="STEP 7: project the GPS log into metric "
                             "coordinates (ENU or UTM) -> gps_metric.csv.")
    cg.add_argument("--gps-file", default=None,
                    help="GPS log CSV (default: gps.file from config).")
    cg.add_argument("--output-dir", default=None,
                    help="output dir for gps_metric.csv (default: paths.georef).")
    cg.add_argument("--crs", default=None, choices=["auto", "enu", "utm"],
                    help="override gps.local_crs.")
    cg.add_argument("--utm-zone", type=int, default=None,
                    help="override gps.utm_zone (1..60).")
    cg.add_argument("--config", default=None, help="run config YAML (default.yaml + merge).")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging(level="WARNING")  # keep CLI output clean; stages configure their own
    args = build_parser().parse_args(argv)
    handlers = {"version": cmd_version, "doctor": cmd_doctor, "init": cmd_init,
                "extract-frames": cmd_extract_frames,
                "select-keyframes": cmd_select_keyframes,
                "calibrate": cmd_calibrate,
                "reconstruct-poses": cmd_reconstruct_poses,
                "show-trajectory": cmd_show_trajectory,
                "convert-gps": cmd_convert_gps}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
