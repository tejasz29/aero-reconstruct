"""Project-root resolution and canonical directory layout.

``PROJECT_ROOT`` is derived from this file's location, so every stage can
build absolute paths without depending on the current working directory::

    from src.common.paths import PROJECT_ROOT, project_paths

    paths = project_paths()
    print(paths.frames)   # <root>/data/frames
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# src/common/paths.py -> parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """All canonical project directories (mirrors the README layout)."""

    root: Path
    raw_videos: Path
    frames: Path
    gps: Path
    imu: Path
    reference: Path
    calibration: Path
    models_depth: Path
    models_segmentation: Path
    outputs: Path
    pointcloud: Path
    mesh: Path
    textures: Path
    reports: Path
    trajectory: Path
    georef: Path
    configs: Path

    def ensure_dirs(self) -> "ProjectPaths":
        """Create every directory (idempotent). Returns self for chaining."""
        for field in self.__dataclass_fields__:
            if field == "root":
                continue
            Path(getattr(self, field)).mkdir(parents=True, exist_ok=True)
        return self

    def missing_dirs(self) -> list[Path]:
        """Directories from the layout that do not exist yet."""
        return [Path(getattr(self, f)) for f in self.__dataclass_fields__
                if f != "root" and not Path(getattr(self, f)).is_dir()]


def project_paths(root: str | Path | None = None) -> ProjectPaths:
    """Build the canonical layout, optionally rooted elsewhere (tests)."""
    r = Path(root) if root is not None else PROJECT_ROOT
    data = r / "data"
    outputs = r / "outputs"
    models = r / "models"
    return ProjectPaths(
        root=r,
        raw_videos=data / "raw_videos",
        frames=data / "frames",
        gps=data / "gps",
        imu=data / "imu",
        reference=data / "reference",
        calibration=r / "calibration",
        models_depth=models / "depth",
        models_segmentation=models / "segmentation",
        outputs=outputs,
        pointcloud=outputs / "pointcloud",
        mesh=outputs / "mesh",
        textures=outputs / "textures",
        reports=outputs / "reports",
        trajectory=outputs / "trajectory",
        georef=outputs / "georef",
        configs=r / "configs",
    )
