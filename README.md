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
python -m src.cli extract-frames --video data/raw_videos/flight.mp4
                             # STEP 2: thinned frames + timestamps.csv -> data/frames/
python -m src.cli select-keyframes
                             # STEP 3: frame_scores.csv + keyframes.csv in data/frames/
python -m src.cli calibrate
                             # STEP 4: calibration/camera.yaml from provided intrinsics,
                             #         checkerboard photos, or Charuco photos
                             #   + report = outputs/reports/calibration_report.json
python -m src.cli reconstruct-poses
                             # STEP 5: camera poses per keyframe via classical SfM
                             #   (SIFT/ORB -> matching -> essential matrix -> chain)
                             #   COLMAP requested (sfm.backend) falls back to the
                             #   bundled OpenCV tracker with a warning
                             #   outputs/trajectory/camera_poses.csv
                             #   + outputs/reports/trajectory_report.json
python -m src.cli show-trajectory
                             # STEP 6: plot camera_poses.csv -> 3D + top-down PNGs
                             #   in outputs/reports/ (headless, matplotlib Agg)
                             #   sanity-check the path before heavy steps
python -m src.cli convert-gps
                             # STEP 7: GPS log -> outputs/georef/gps_metric.csv
                             #   + outputs/reports/gps_report.json (ENU/UTM)
python -m src.cli align-trajectory
                             # STEP 8: fit the similarity Xg = s*R*Xv + t that
                             #   maps the SfM trajectory onto gps_metric.csv
                             #   (RANSAC + closed-form refinement, RTK-aware)
                             #   resolves the monocular scale
                             #   outputs/georef/aligned_trajectory.csv
                             #   + reports/aligned_trajectory_{transform,report}.json
                             #   exits 1 when the fit is rejected
python -m src.cli predict-depth
                             # STEP 9: relative monocular depth + confidence
                             #   per keyframe (dummy CPU backend by default,
                             #   HF Depth-Anything when available)
                             #   outputs/depth/depth_*.npy + confidence_*.npy
                             #   + depth_index.csv + reports/depth_report.json
                             #   depth is NOT metric until STEP 10
python -m src.cli unproject-depth
                             # STEP 10: depth -> per-frame + merged colored
                             #   clouds (metric-via-GPS-scale, NOT absolute CRS)
                             #   outputs/pointcloud/scene.ply + frame_*.ply
                             #   + cloud_index.csv + reports/unproject_report.json
```

Pipeline-stage subcommands (`preprocess`, `reconstruct`, …) are
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
(STEP 2) frame extraction math, (STEP 3) quality scorers + keyframe selection,
(STEP 4) intrinsics model + checkerboard/Charuco + expected-output runner tests,
(STEP 5) SIFT/ORB extraction + matching, essential-matrix/PnP recovery on
synthetic GT, pose chaining, runner e2e on a rendered 5-view pass, rejection
and resume paths, backend fallback.
(STEP 6) pose CSV round-trip, camera-axis convention, top-down projection math,
headless PNG output incl. single-pose and rejected-only edge cases, CLI surface.
(STEP 7) GPS reader + column aliases, WGS84 bounds validation, geodesic
distance ground truth, ENU/UTM conversions, CRS auto-resolution, runner e2e.
(STEP 8) transform contract + JSON round-trip, Umeyama recovery of a known
`s/R/t` on a 3D path, planar (nadir) fit and its documented limit, degeneracy
refusal (collinear/mirrored), RANSAC outlier rejection with named outliers,
pose/GPS association and the time gate, RTK tier handling + accuracy wording,
metric-CSV read-back, runner e2e (3D, planar, RTK, rejected) and the CLI
surface including its non-zero exit on a rejected fit.
(STEP 9) preprocess resize/normalize, confidence range [0,1] + validation,
dummy-backend shapes/finite/determinism, predict_single full-res output,
depth stats, .npy/preview/index/report round-trips, runner e2e on synthetic
keyframes + tiny-image determinism, CLI surface.
(STEP 10) pinhole unproject + reproject round-trip, pose/scale correctness,
colored-cloud color/stride/invalid/determinism, PLY/index/report round-trips,
runner e2e (2-frame synthetic, merged + per-frame, scale flag) + CLI surface.

Current total: **230 tests passing**.

## 6. Pipeline status

| STEP | Stage | State |
|------|-------|-------|
| 1 | Project structure + environment | ✅ done |
| 2 | Video loading + frame extraction | ✅ done |
| 3 | Frame quality + keyframe selection | ✅ done |
| 4 | Camera calibration | ✅ done |
| 5 | SfM baseline (poses) | ✅ done (classical OpenCV; COLMAP fallback) |
| 6 | Trajectory visualisation | ✅ done |
| 7 | GPS parsing + metric conversion | ✅ done |
| 8 | Visual ↔ GPS alignment | ✅ done (RANSAC + Umeyama refinement, RTK-aware) |
| 9 | Learned depth inference | ✅ done (dummy CPU + HF fallback, relative only) |
| 10 | Depth → 3D | ✅ done (per-frame + merged, metric-via-scale) |
| 11–12 | Fusion + filtering | ⬜ |
| 13 | Dynamic-object segmentation | ⬜ |
| 14–15 | Mesh + texture | ⬜ |
| 16–17 | Georeferencing + evaluation | ⬜ |
| 18–19 | Viewer + backend | ⬜ |
| 20 | Near-real-time optimisation | ⬜ (after offline works) |

**Next recommended step: STEP 11** — point-cloud fusion
(`src/fusion/`): merge per-frame clouds with confidence weighting, voxel
downsample, duplicate removal. STEP 10 clouds are metric-via-scale inputs.
