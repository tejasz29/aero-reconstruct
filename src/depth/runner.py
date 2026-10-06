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


def _read_keyframes(keyframes_csv) -> list[dict]:
    """Read keyframes.csv (frame_id, filename); raises FileNotFoundError."""
    import csv
    from pathlib import Path

    p = Path(keyframes_csv)
    if not p.is_file():
        raise FileNotFoundError(f"keyframes file not found: {p} "
                                "(run select-keyframes first)")
    with open(p, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"no keyframes in {p}")
    return rows


def run_depth_prediction(cfg: dict, frames_dir=None, keyframes_file=None,
                         output_dir=None, backend=None, device=None,
                         input_size=None) -> dict:
    """STEP 9 entry point: predict relative depth + confidence per keyframe."""
    from pathlib import Path

    import numpy as np

    from src.depth.inference import depth_stats, predict_single
    from src.depth.io import (save_confidence_npy, save_depth_npy,
                              save_preview_png, write_depth_index,
                              write_depth_report)
    from src.depth.model import get_backend, resolve_device
    from src.depth.preprocess import load_image
    from src.depth.types import RELATIVE_DEPTH_NOTE

    paths = resolve_paths(cfg, frames_dir, keyframes_file, output_dir)
    policy = resolve_policy(cfg, backend=backend, device=device,
                            input_size=input_size)
    rows = _read_keyframes(paths.keyframes_csv)
    concrete = resolve_device(policy.device)
    active = get_backend(policy.backend, model_id=policy.model,
                         device=concrete, seed=policy.seed)
    log.info("predicting depth for %d keyframes with backend=%s on %s",
             len(rows), active.name, concrete)

    out_dir = Path(paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict] = []
    for row in rows:
        fid = int(row.get("frame_id", len(index_rows)))
        filename = str(row.get("filename", row.get("file", f"frame_{fid:06d}.jpg")))
        img_path = Path(paths.frames_dir) / filename
        image = load_image(img_path)
        h, w = image.shape[:2]
        depth, conf = predict_single(image, active,
                                     input_size=policy.input_size)
        stats = depth_stats(depth)
        stem = Path(filename).stem
        depth_path = save_depth_npy(depth, out_dir / f"depth_{stem}.npy")
        conf_path = ""
        if policy.store_confidence:
            conf_path = str(save_confidence_npy(conf, out_dir / f"confidence_{stem}.npy"))
        save_preview_png(depth, out_dir / f"depth_{stem}.png")
        index_rows.append({
            "frame_id": fid, "filename": filename, "width": h and w and w,
            "height": h, "depth_path": str(depth_path),
            "confidence_path": conf_path,
            "depth_min": round(stats["min"], 6),
            "depth_max": round(stats["max"], 6),
            "depth_mean": round(stats["mean"], 6),
            "confidence_mean": round(float(np.mean(conf)), 6),
        })
    index_csv = str(write_depth_index(index_rows, paths.index_csv))
    report = {"backend": active.name, "model": policy.model, "device": concrete,
              "policy": policy.as_dict(), "n_keyframes": len(index_rows),
              "index_csv": index_csv, "relative": True,
              "note": RELATIVE_DEPTH_NOTE}
    report_json = str(write_depth_report(report, paths.report_json))
    log.warning("depth is RELATIVE, not metric — %s", RELATIVE_DEPTH_NOTE)
    return {"policy": policy, "backend": active.name, "device": concrete,
            "n_keyframes": len(index_rows), "index_csv": index_csv,
            "report_json": report_json, "rows": index_rows}
