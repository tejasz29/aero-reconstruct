"""Command-line entry point for single-pass-3d.

Usage (from the project root)::

    python -m src.cli version
    python -m src.cli doctor
    python -m src.cli init

or, after ``pip install -e .``::

    sp3d version | doctor | init

* ``version`` — print the package version.
* ``doctor``  — verify interpreter, folder layout, FFmpeg and optional
  heavy dependencies; exits non-zero if the core layout is broken.
* ``init``    — create any missing project directories (idempotent).

Pipeline-stage subcommands (preprocess, calibrate, reconstruct, …) are
added in later STEPS; this file only scaffolds STEP 1.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys

from src.common.config_loader import get, load_config
from src.common.logging_utils import get_logger, setup_logging
from src.common.paths import PROJECT_ROOT, project_paths

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sp3d",
        description="Single-pass drone video -> 3D reconstruction (offline MVP).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print version and exit.")
    sub.add_parser("doctor", help="check environment and project layout.")
    sub.add_parser("init", help="create missing project directories.")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging(level="WARNING")  # keep CLI output clean; stages configure their own
    args = build_parser().parse_args(argv)
    handlers = {"version": cmd_version, "doctor": cmd_doctor, "init": cmd_init}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
