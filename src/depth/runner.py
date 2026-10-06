"""STEP 9 — runner: keyframes -> relative depth + confidence maps.

Inputs:  ``data/frames/keyframes.csv`` (STEP 3 survivors) + images.
Outputs: ``outputs/depth/depth_*.npy`` + ``confidence_*.npy`` (+ previews)
         ``outputs/depth/depth_index.csv``
         ``outputs/reports/depth_report.json``

Depth is NOT metric: the runner stamps every artefact with the
relative-depth note so STEP 10 (unprojection) must apply the STEP 8 scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.logging_utils import get_logger

log = get_logger("sp3d.depth")


@dataclass(frozen=True)
class DepthPolicy:
    """Every tunable depth inference obeys, resolved from config."""

    backend: str = "dummy"
    model: str = "depth-anything-v2"
    device: str = "auto"
    input_size: int = 518
    store_confidence: bool = True
    seed: int = 42

    def as_dict(self) -> dict:
        return {"backend": self.backend, "model": self.model,
                "device": self.device, "input_size": self.input_size,
                "store_confidence": self.store_confidence, "seed": self.seed}


def resolve_policy(cfg: dict, backend: str | None = None,
                   device: str | None = None,
                   input_size: int | None = None) -> DepthPolicy:
    """Read the ``depth.*`` section of the config into a policy."""
    from src.common.config_loader import get

    defaults = DepthPolicy()
    return DepthPolicy(
        backend=str(backend or get(cfg, "depth.backend",
                                   get(cfg, "depth.model", defaults.backend))),
        model=str(get(cfg, "depth.model", defaults.model)),
        device=str(device or get(cfg, "depth.device", defaults.device)),
        input_size=int(input_size if input_size is not None
                       else get(cfg, "depth.input_size", defaults.input_size)),
        store_confidence=bool(get(cfg, "depth.store_confidence",
                                  defaults.store_confidence)),
        seed=int(get(cfg, "project.seed", defaults.seed)),
    )


@dataclass(frozen=True)
class DepthPaths:
    """Input and output locations of one depth run."""

    frames_dir: object
    keyframes_csv: object
    output_dir: object
    index_csv: object
    report_json: object


def resolve_paths(cfg: dict, frames_dir=None, keyframes_file=None,
                  output_dir=None) -> DepthPaths:
    """Resolve keyframes input + outputs/depth + reports locations."""
    from pathlib import Path

    from src.common.config_loader import get
    from src.common.paths import PROJECT_ROOT

    def _abs(p) -> Path:
        c = Path(p)
        return c if c.is_absolute() else PROJECT_ROOT / c

    frames = Path(frames_dir) if frames_dir else _abs(
        get(cfg, "paths.frames", "data/frames"))
    keyframes = Path(keyframes_file) if keyframes_file else frames / "keyframes.csv"
    out = Path(output_dir) if output_dir else _abs("outputs/depth")
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    reports = _abs(get(cfg, "paths.reports", "outputs/reports"))
    return DepthPaths(frames_dir=frames, keyframes_csv=keyframes,
                      output_dir=out, index_csv=out / "depth_index.csv",
                      report_json=reports / "depth_report.json")
