# single-pass-3d — Single-Pass Drone Video → AI 3D Reconstruction (offline MVP)

Takes **one continuous drone video** from a single UAV flight pass (+ GPS/flight
metadata) and produces a **georeferenced, metrically useful 3D representation**:
terrain, buildings, facades, rooftops, roads, vegetation, colored point cloud,
textured mesh.

> **Accuracy rules (non-negotiable for this project)**
> - **Relative accuracy** (shape quality) ≠ **absolute accuracy** (real-world
>   position/scale). Report both, separately.
> - Ordinary GPS does **not** justify centimeter-level claims. cm-level absolute
>   accuracy requires RTK/PPK or surveyed reference data.
> - Unseen/occluded surfaces are **never** claimed as measured — AI-completed
>   geometry is labelled *estimated* and confidence is preserved end-to-end.

Classical baseline first (COLMAP SfM + GPS alignment), then AI depth, then
segmentation — every stage independently testable, every intermediate output
saved.

---

## 1. Repository layout

```text
data/            raw_videos/ frames/ gps/ imu/ reference/   (inputs — never modified)
calibration/     camera.yaml                                (intrinsics)
models/          depth/ segmentation/                       (weights — not committed)
src/             video/ calibration/ tracking/ sfm/ depth/ fusion/
                 segmentation/ georef/ mesh/ texture/ evaluation/ common/
outputs/         pointcloud/ mesh/ textures/ reports/ trajectory/ georef/
api/             FastAPI backend (STEP 19)
frontend/        React/Next.js + Three.js viewer (STEP 18)
tests/           unit tests (geometry + coordinates tests arrive with their stages)
configs/         default.yaml  logging.yaml                 (all tunables live here)
```

Raw inputs under `data/` are **read-only** for the pipeline. Everything generated
goes under `outputs/`.

## 2. Setup

**Target interpreter: Python 3.10 / 3.11.** CUDA-capable NVIDIA GPU preferred
for depth/segmentation; CPU works for the classical stages.

```powershell
# option A — venv
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt                 # CPU torch by default
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA (first)

# option B — conda (includes ffmpeg + gdal)
conda env create -f environment.yml
conda activate single-pass-3d

# dev extras (pytest, ruff) + editable install for the `sp3d` command
pip install -r requirements-dev.txt
pip install -e .
```

FFmpeg must be on `PATH` (needed from STEP 2). COLMAP is needed from STEP 5.

## 3. CLI

```powershell
python -m src.cli version    # or: sp3d version
python -m src.cli doctor     # env + layout + dependency health check
python -m src.cli init       # (re)create missing project directories
```

Pipeline-stage subcommands (`preprocess`, `calibrate`, `reconstruct`, …) are
added in their respective steps.

## 4. Configuration

`configs/default.yaml` holds **every** tunable (16 sections, one per stage) —
no hard-coded values in `src/`. Override per run without editing the base file:

```powershell
$env:SP3D_CONFIG = "configs/my_run.yaml"   # deep-merged over default.yaml
```

Logging (`configs/logging.yaml`): one format, console + rotating
`outputs/reports/run.log`, configured once via
`src.common.logging_utils.setup_logging()`.

## 5. Testing

```powershell
python -m pytest tests/ -v
```

Current coverage (STEP 1): layout integrity, config load/merge/`get()`,
logging (file + rotation + YAML-driven), CLI smoke tests.
Geometry/coordinate-conversion unit tests are added with STEPS 4–8.

## 6. Pipeline status

| STEP | Stage | State |
|------|-------|-------|
| 1 | Project structure + environment | ✅ done |
| 2 | Video loading + frame extraction | ⬜ next |
| 3 | Frame quality + keyframe selection | ⬜ |
| 4 | Camera calibration | ⬜ |
| 5 | COLMAP/SfM baseline | ⬜ |
| 6 | Trajectory visualisation | ⬜ |
| 7 | GPS parsing + metric conversion | ⬜ |
| 8 | Visual ↔ GPS alignment | ⬜ |
| 9–10 | Learned depth → 3D | ⬜ |
| 11–12 | Fusion + filtering | ⬜ |
| 13 | Dynamic-object segmentation | ⬜ |
| 14–15 | Mesh + texture | ⬜ |
| 16–17 | Georeferencing + evaluation | ⬜ |
| 18–19 | Viewer + backend | ⬜ |
| 20 | Near-real-time optimisation | ⬜ (after offline works) |

**Next recommended step: STEP 2** — video loading + frame extraction
(`src/video/`): FPS detection, timestamp extraction, frame dump,
`timestamps.csv`. No quality scoring yet (that's STEP 3).
