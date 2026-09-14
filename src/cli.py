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
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging(level="WARNING")  # keep CLI output clean; stages configure their own
    args = build_parser().parse_args(argv)
    handlers = {"version": cmd_version, "doctor": cmd_doctor, "init": cmd_init,
                "extract-frames": cmd_extract_frames,
                "select-keyframes": cmd_select_keyframes}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
