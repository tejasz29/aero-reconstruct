# single-pass-3d — Full Implementation Steps  session id-ses_f6181e32effeQYcoluE1IoxOJA

Source of truth for build order. A step is **done** only when: implemented,
tested green, demoed on real or synthetic data, README status updated, and
committed + pushed. Never start the next step on a broken tree.

**Progress: STEPS 1–10 done · STEP 11 next · 230/230 tests passing.**

Conventions every step follows: tunables live in `configs/default.yaml`
(never hard-coded); every stage logs via `src.common.logging_utils`;
raw inputs under `data/` are read-only; all artefacts go under `outputs/`;
each step adds a CLI subcommand and unit tests.

---

## STEP 1 — Project setup ✅ done

**What:** Empty repo → runnable scaffold: folder layout, environment files,
configs, logging, CLI skeleton, test harness. No pipeline logic.

**Files:** full `data/src/outputs/api/frontend/tests/configs` tree ·
`requirements.txt`, `requirements-dev.txt`, `environment.yml`,
`pyproject.toml`, `.gitignore` · `configs/default.yaml` (16 sections),
`configs/logging.yaml`, `calibration/camera.yaml` (placeholder) ·
`src/common/{logging_utils,config_loader,paths}.py` · `src/cli.py`
(`version`/`doctor`/`init`) · `tests/{conftest,test_cli,test_config,
test_logging,test_project_layout}.py` · `README.md`.

**Tests:** 15 passed (layout, config load/merge, logging incl. file output, CLI).
**Verify:** `python -m pytest tests/` · `python -m src.cli doctor` → OK.

**Commits (10):** `b965853` structure placeholders · `c55af55` runtime deps ·
`3ce8d74` dev env + packaging + gitignore · `f3af9c8` master config ·
`abb7f5e` logging config + camera stub · `ff7edd4` src/api/frontend scaffold ·
`c57f795` paths utility · `5d4eb53` config loader · `01cb478` logging setup ·
`3f154be` CLI + tests + README.

---

## STEP 2 — Video loading + frame extraction ✅ done

**What:** Probe the drone video (FPS, resolution, frame count, codec) and dump
an evenly thinned frame set + `timestamps.csv`. No quality judgements yet.

**Algorithm:** stride = round(src_fps / target_fps) → uniform-subsample to
`max_frames` → sequential decode → downscale beyond `image_max_width` → save.
Timestamps are `source_index / fps` (exact for CFR; VFR caveat documented).

**Files:** `src/video/video_info.py` (`probe_video`, `frame_timestamp`) ·
`src/video/frame_extractor.py` (`compute_sample_indices`, `extract_frames`,
`read_timestamps_csv`) · `src/cli.py` += `extract-frames` · config +=
`video.output_format`, `video.jpeg_quality`.

**Outputs:** `data/frames/frame_000000.jpg…` + `timestamps.csv`
(`frame_id,source_index,timestamp_s,filename,width,height`).
**Tests:** 26 passed (sampling math, probing, resize, CSV, CLI; synthetic MP4s
generated on the fly — no real footage needed).
**Verify:** `python -m src.cli extract-frames --video data/raw_videos/<f>.mp4`
→ check `data/frames/`.

**Commits (6):** `a90f09a` probing · `b43f8e9` extraction · `0fc5c6b` API export ·
`e5c130b` CLI · `88fe51b` tests · `b3b1f1e` config + README.

---

## STEP 3 — Frame quality + keyframe selection ✅ done

**What:** Score every extracted frame (blur, exposure, feature count), drop bad
frames + near-duplicates, keep an auditable verdict per frame.

**Algorithm:** blur = Laplacian variance · exposure = mean gray ·
features = ORB count (SIFT optional) · duplicates = dHash Hamming distance.
Gates in exposure → blur → features order, then greedy time-gap + dedup pass,
then uniform cap at `max_keep`. `min_translation_m` deferred to STEP 8 (needs
metric scale).

**Files:** `src/video/quality.py` · `src/video/keyframes.py`
(`score_frames`, `select_keyframes`, `run_selection`) · `src/cli.py` +=
`select-keyframes` · config += `video.quality.detector`,
`video.keyframes.dedup_hamming_threshold`.

**Outputs:** `data/frames/frame_scores.csv` (every frame + keep/reject reason)
+ `data/frames/keyframes.csv` (survivors — input to STEPS 4–5).
**Tests:** 39 passed (scoring, gates, time-gap, dedup, max-keep, e2e, CLI).
**Verify:** `python -m src.cli select-keyframes` → inspect reject-reason
summary and both CSVs.

**Commits (15):** `6aaffb3` csv reader · `5975c4b` blur · `cc04230` exposure ·
`8ceea17` features · `b3c431e` dhash · `a89c3f2` result types · `72e66fb`
scoring · `8a7578d` gates + dedup · `3555712` writers + run · `bdb2beb` API ·
`af1338d` scoring tests · `5400134` selection tests · `1b43b10` e2e tests ·
`af0219c` CLI · `c3c67e1` config + README.

---

## STEP 4 — Camera calibration ✅ done

**What:** Fill `calibration/camera.yaml` with real intrinsics (fx, fy, cx, cy,
distortion).
**Inputs:** vendor intrinsics if provided (`calibration.source=provided`),
else calibration photos — checkerboard or Charuco board.
**Algorithm:** `Intrinsics` model is the canonical K (`camera_matrix()`),
loaded/saved in YAML, validated (positive focal, sane principal point). Board
solvers use `cv2.calibrateCamera` (checkerboard: `findChessboardCornersSB`;
Charuco: `CharucoDetector` + `getChessboardCorners` id indexing, plain
`calibrateCamera` fallback for envs with `calibrateCameraCharuco`). A runner
(`resolve_calibration`) picks the source and writes `camera.yaml`, a JSON
report, and annotated views; reprojection error vs threshold verdict.

**Files:** `src/calibration/{intrinsics,checkerboard,charuco,runner}.py` ·
`src/cli.py` += `calibrate` · config += `calibration.provided_file`,
`calibration.images_dir`, `calibration.checkerboard`, `calibration.charuco`,
`calibration.validation.max_reprojection_error_px`, `paths.reports`.

**Outputs:** `calibration/camera.yaml` (intrinsics + distortion + RMS) ·
`outputs/reports/calibration_report.json` + annotated views under
`outputs/reports/calibration/`.
**Tests:** 53 passed (+14: intrinsics model round-trip/validate/undistort,
checkerboard detect + ground-truth calibration recover K, needed-min-views,
Charuco detect + calibration on rendered boards, runner e2e for all three
sources incl. unknown-source error).
**Verify:** `python -m src.cli calibrate` (config-driven; source & paths in
`configs/default.yaml` or a run-config YAML) → check `camera.yaml` + report.

**Commits:** STEP 4 commits listed after push (see git log).

## STEP 5 — SfM baseline (poses) ✅ done

**What:** Camera poses for every keyframe. COLMAP kept as the named backend with
an automatic fallback to a bundled classical OpenCV tracker (`sfm.backend`,
resolved at runtime; no COLMAP binary needed for the demo).
**Inputs:** `keyframes.csv` + intrinsics. **Outputs:**
`outputs/trajectory/camera_poses.csv` (frame id, timestamp, position, rotation,
inliers, mean reproj error, median parallax) + `outputs/reports/
trajectory_report.json`; pose rejected above thresholds with an auditable
`reject_reason`.
**Algorithm:** features per frame (SIFT or ORB) → reciprocal-consistent
matching with Lowe's ratio test → RANSAC essential matrix over undistorted
pixels → cheirality-validated (triangulated points in front of both cameras)
hypothesis with 5-dof Gauss-Newton refinement on the consensus → PnP support
(`estimate_absolute_pose`) → sequential chaining (world = first accepted
keyframe, anchor stays put on rejection so the run resumes). Reject reasons:
`no_features`, `match_failure`, `low_inliers`, `degenerate`,
`high_reprojection`, `low_parallax`.
**Scale caveat (documented):** monocular motion is recovered up to one global
scale (essential matrices return unit-norm translation) — resolved to metric
coordinates by the visual↔GPS similarity alignment in STEP 8. The *sign* of
translation is ambiguous for near-zero rotation / weak 3D parallax; the chain
enforces temporal consistency via the previous accepted baseline and the GPS
alignment decides the absolute handedness.
**Files:** `src/sfm/{features,pose,runner,__init__}.py` · `src/cli.py` +=
`reconstruct-poses` · config += `paths.trajectory`, `sfm.backend`,
`sfm.feature`, `sfm.max_features`, `sfm.matcher_ratio_test`,
`sfm.ransac_reproj_threshold_px`, `sfm.max_reprojection_error_px`,
`sfm.min_inliers`, `sfm.min_parallax_px`.
**Tests:** 65 passed (+12: relative-pose GT recovery, degenerate rejection,
prior-sign tie-break, PnP recovery + min-points, chain math, rendered-view
feature/matching, runner e2e on a synthetic 5-view textured pass with 3D
structure, featureless-frame rejection + resume, CSV schema, colmap→opencv
fallback, CLI parser).
**Verify:** `python -m src.cli reconstruct-poses` (config-driven) → check
`outputs/trajectory/camera_poses.csv` + report; COLMAP requested ⇒ warning +
OpenCV tracker used.

**Commits:** STEP 5 commits listed after push (see git log).

## STEP 6 — Trajectory visualisation ✅ done

**What:** Sanity-check the STEP 5 path (continuity, orientation, rejected
frames) before any heavy downstream step burns GPU hours.
**Inputs:** `outputs/trajectory/camera_poses.csv` (STEP 5).
**Outputs:** `outputs/reports/trajectory_3d.png` + `trajectory_topdown.png`
(matplotlib, headless Agg — no display needed).
**Algorithm:** `read_poses_csv` parses the STEP 5 CSV back into `CameraPose`s;
`plot_trajectory` renders a 3D view (path + world-origin star + per-pose
orientation frustums from the world->camera axes) and a 2D top-down view
(selectable projection `xy|xz|yz` with forward-facing arrows), annotating the
accepted/rejected summary and mean reprojection error. No accepted poses ⇒
clear `ValueError`.
**Files:** `src/sfm/visualize.py` · `src/cli.py` += `show-trajectory` ·
config += `visualization.dpi`, `visualization.figsize`,
`visualization.topdown_projection`, `visualization.draw_frustums`,
`visualization.frustum_scale`.
**Tests:** 79 passed (+9: pose-CSV round-trip, camera-axis convention,
top-down projection math, headless PNG output for 5/1/0-kept and rejected-only
poses, unknown-projection error, CLI parser).
**Verify:** `python -m src.cli show-trajectory` (config-driven; `--poses-csv`,
`--output-dir`, `--topdown` overrides) → inspect `outputs/reports/trajectory_*.png`.

**Commits:** STEP 6 commits listed after push (see git log).

## STEP 7 — GPS parsing + metric conversion ✅ done

**What:** `timestamp,lat,lon,alt` → metric coordinates. Local ENU for small
scenes, auto UTM for larger ones. Never treat lat/lon as metres.
**Algorithm:** `GPSFix` contract (`timestamp_s,latitude,longitude,
altitude_m`, WGS84) → CSV reader with column aliases (time/t, lat, lng/alt,
…) → range validation (lat ±90, lon ±180, alt ≥ -500 m, finite timestamps) →
`wgs84_distance_m` (pyproj `Geod`) → ENU tangent plane from pure ECEF math
(Bowring-style radius of curvature, closed-form rotation anchored at the
first fix) or UTM via pyproj `Transformer` (zone `1..60`, EPSG 326xx/327xx,
heights relative to the first fix). `resolve_crs` honours `gps.local_crs`
(`auto|enu|utm`): `auto` stays ENU while the track extent
(`track_extent_m`, bounding-box diagonal) is under `gps.auto_crs_max_extent_m`,
otherwise switches to UTM. A runner (`run_gps_conversion`) reads config,
projects, and writes both outputs.
**Outputs:** `outputs/georef/gps_metric.csv`
(`timestamp_s,latitude,longitude,altitude_m,easting_m,northing_m,up_m,crs,zone`)
+ `outputs/reports/gps_report.json` (source, n_fixes, crs, zone, WGS84 datum,
origin, extent_m).
**Files:** `src/georef/gps.py` · `src/georef/__init__.py` · `src/cli.py` +=
`convert-gps` (`--gps-file`, `--output-dir`, `--crs`, `--utm-zone`) · config
+= `gps.file`, `gps.local_crs`, `gps.auto_crs_max_extent_m`, `gps.utm_zone`.
**Tests:** 96 passed (+16: fix contract, canonical + aliased CSV headers,
missing-column error, WGS84 bounds accept/reject, Eiffel→Louvre geodesic
ground truth (~3177 m), ENU origin + direction/scale, UTM zone/EPSG lookup,
UTM projection plausibility, track extent, CRS auto/ENU/UTM rules + override,
runner e2e outputs, CLI parser).
**Verify:** `python -m src.cli convert-gps` (config-driven; flags override)
→ check `outputs/georef/gps_metric.csv` + `outputs/reports/gps_report.json`.

**Commits (33):** STEP 7 split into 33 granular commits, pushed to origin/main
(see `git log --oneline 1d5e026..HEAD`).

## STEP 8 — Visual ↔ GPS alignment ✅ done

**What:** Resolve the SfM scale ambiguity: `X_global ≈ s·R·X_visual + t` via
robust similarity alignment (RANSAC + least-squares refinement). RTK/PPK path
when available.
**Algorithm:** pair accepted poses with metric fixes by nearest timestamp
(one-to-one, `alignment.max_time_gap_s` gate) → keep only RTK/PPK-fixed
fixes when the log carries a fix-status column and tighten the residual gate
from 2 m to 0.5 m → RANSAC over minimal samples scored by inlier count (ties
broken by median residual) → re-fit the winners in closed form (Umeyama 3D,
or a yaw-only planar solve for a nadir pass) → re-evaluate the inlier mask →
report RMSE/median/max. Degenerate paths (coincident or collinear centres)
and too few inliers are *refused* with an auditable `reject_reason`, and a
refused fit carries the identity transform so nothing downstream can
georeference with it. `mode: 2d` is exact for a flat nadir pass and wrong for
a tilted one — the residual is what reveals it.
**Accuracy policy:** the reported RMSE describes *how well the trajectory was
aligned to the GPS track*. `accuracy_summary` states, per run, whether the
fixes were RTK/PPK-fixed, plain consumer GPS, or unknown, because a residual
against metre-level GPS is not a centimetre-level absolute-accuracy claim.
**Outputs:** `outputs/georef/aligned_trajectory.csv` (one row per STEP 5
pose: visual centre, aligned centre, matched GPS target, `dt`, inlier flag,
residual, re-expressed rotation) ·
`outputs/reports/aligned_trajectory_transform.json` (the `s/R/t` contract for
STEPS 10/16 and the viewer) · `outputs/reports/aligned_trajectory_report.json`
(inputs, policy, inlier stats, pairing stats, GPS tier, accuracy wording).
**Files:** `src/georef/align.py` (transform contract, Umeyama + planar
solvers, residuals, RANSAC, association, RTK policy, rows/CSV/JSON writers) ·
`src/georef/runner.py` (config policy + path resolution, pairing, fit,
artefacts, rejection paths) · `src/georef/gps.py` += `read_metric_table` /
`read_metric_csv` · `src/cli.py` += `align-trajectory` · config +=
`alignment.*` (mode, ransac_iterations, inlier_threshold_m,
min_correspondences, min_inliers, max_time_gap_s, rtk_inlier_threshold_m,
use_rtk_if_available, output_name).
**Tests:** 180 passed (+84: transform contract/serialisation, known `s/R/t`
recovery on a 3D path, planar fit + its tilt limit, collinear/mirrored
refusal, RANSAC outlier rejection with named outliers and determinism,
refinement improvement, pose/GPS association + time gate, fix-status
normalisation + RTK selection + accuracy wording, metric-CSV read-back,
runner e2e in 3D/planar/RTK/rejected modes, CLI surface and exit code).
**Verify:** `python -m src.cli align-trajectory` (config-driven; `--poses-csv`,
`--gps-metric-csv`, `--output-dir`, `--mode`, `--inlier-threshold` override)
→ check `outputs/georef/aligned_trajectory.csv` + both JSON artefacts; a
rejected fit prints the reason and exits 1.

**Commits (40):** STEP 8 split into granular commits (association, fit,
refinement, RANSAC, RTK, accuracy, writers, runner, CLI, tests) — see
`git log --oneline origin/main..HEAD`.

## STEP 9 — Learned depth inference ✅ done

**What:** Monocular depth (+ confidence) per keyframe. Dummy CPU backend by
default (deterministic, no weights) with an automatic fallback from the
Depth Anything HF backend when torch/transformers or weights are missing.
Depth is NOT metric — constrained later by SfM/GPS scale in STEP 10.
**Inputs:** `data/frames/keyframes.csv` + images (STEP 3).
**Outputs:** `outputs/depth/depth_*.npy` + `confidence_*.npy` (+ grayscale
previews) · `outputs/depth/depth_index.csv` (one row per keyframe) ·
`outputs/reports/depth_report.json` (backend, policy, relative-depth note).
**Algorithm:** PIL load → longest-side resize (`depth.input_size`, aspect
kept, 2x upscale cap) → backend predict at model size → bilinear resize back
to HxW → `ensure_finite` (median fill) → edge-aware confidence
`1/(1+|grad|/median)` in [0,1]. `resolve_device` maps `auto|cuda|cpu`;
`get_backend` tries HF then warns + falls back to dummy.
**Files:** `src/depth/{types,preprocess,confidence,model,inference,io,
runner}.py` · `src/cli.py` += `predict-depth` · config += `depth.backend`,
`depth.input_size`, `depth.store_confidence` (plus existing model/device).
**Tests:** 208 passed (+28: preprocess resize/normalize/load, confidence
range/flat/uniform/validation, device/backend/dummy determinism, ensure_finite,
predict_single shapes + full-res + determinism, depth stats, npy/preview/
confidence/index/report round-trips, policy/paths, runner e2e (3-keyframe
synthetic pass) + tiny-image determinism smoke test + missing-keyframes
error, CLI parser + e2e).
**Verify:** `python -m src.cli predict-depth --backend dummy` → check
`outputs/depth/depth_index.csv` + report; depth values are relative.

**Commits (32):** STEP 9 split into 32 granular commits (types, preprocess,
confidence, backends, inference, io, runner, CLI, config, tests, docs) — see
`git log --oneline` since the steps 1–8 baseline snapshot.

## STEP 10 — Depth → 3D ✅ done

**What:** Unproject each keyframe (`X=(u-cx)·Z/fx`, …) to camera frame, to
STEP 5 world via poses, to metric via STEP 8 scale; emit colored points.
Per-frame PLY + merged `scene.ply` (user chose per-frame + merged).
Metric-via-scale only — absolute CRS deferred to STEP 16.
**Inputs:** `outputs/depth/depth_index.csv` + images · `calibration/camera.yaml` ·
`outputs/trajectory/camera_poses.csv` · `aligned_trajectory_transform.json` (scale).
**Outputs:** `outputs/pointcloud/frame_*.ply` + `outputs/pointcloud/scene.ply` ·
`outputs/pointcloud/cloud_index.csv` (STEP 11 input) ·
`outputs/reports/unproject_report.json` (`metric_via_gps_scale=true`,
`absolute_crs=false`).
**Algorithm:** vectorized pinhole inverse → `valid_mask` (finite + min/max) →
optional stride thinning → `Xw=R_wc.T@Xcam+C` → `×s` → attach RGB + confidence.
Missing transform or REJECTED status warns + falls back to s=1 (relative).
**Files:** `src/fusion/{types,unproject,io,runner}.py` · `src/cli.py` +=
`unproject-depth` · config += `fusion.unproject_stride`, `min/max_depth_m`,
`save_per_frame`.
**Tests:** 230 passed (+22: pixel grid, unproject/reproject round-trip,
pose identity/rotation, scale contract, valid gates, colored-cloud shapes/
color/stride/invalid/determinism/stats, PLY/index/report round-trips,
policy/paths, runner e2e 2-frame synthetic + missing-transform + missing-input
errors, CLI parser + e2e).
**Verify:** `python -m src.cli unproject-depth --stride 2` → check
`outputs/pointcloud/scene.ply` + `cloud_index.csv` + report.

**Commits (34):** STEP 10 split into 34 granular commits (types, unproject,
pose/scale, cloud, io, runner, CLI, config, tests, docs) — see `git log`.

## STEP 11 — Point-cloud fusion ⬜

**What:** Merge all frames: confidence weighting, voxel downsample, duplicate
removal, normal estimation. **Outputs:** `outputs/pointcloud/scene.ply`
(+ LAS/LAZ). **Files:** `src/fusion/` · CLI `fuse-cloud`.

## STEP 12 — Point-cloud filtering ⬜

**What:** Statistical + radius outlier removal. **Outputs:** cleaned cloud
(side-by-side density/outlier stats before/after). **Files:** `src/fusion/
filter.py`. **Tests:** synthetic outlier cloud → removal rate assertions.

## STEP 13 — Dynamic-object segmentation ⬜

**What:** YOLO-seg (/SAM 2) masks for people/vehicles, applied BEFORE fusion;
compare with/without masking, measure artifact reduction. **Files:**
`src/segmentation/` · CLI `mask-dynamics`.

## STEP 14 — Mesh reconstruction ⬜

**What:** Open3D Poisson baseline from cleaned cloud; cut low-density verts;
preserve gaps, never invent geometry. **Outputs:** `scene_mesh.{ply,obj,glb}`.
**Files:** `src/mesh/` · CLI `build-mesh`.

## STEP 15 — Texture mapping ⬜

**What:** Project sharp, well-exposed, well-angled frames onto visible mesh;
blend overlaps. **Outputs:** textured GLB + OBJ + textures. **Files:**
`src/texture/` · CLI `texture-mesh`.

## STEP 16 — Georeferencing ⬜

**What:** Final model → global CRS (ENU/UTM), store CRS + transform + origin +
alignment error; optional DSM GeoTIFF. **Files:** `src/georef/` extensions.

## STEP 17 — Evaluation metrics ⬜

**What:** Quantitative report only (no visual-only claims): RMSE/MAE, XYZ
georeferencing error, completeness %, density, outlier ratio, artifact
reduction, timings, measurement errors — weighted per the 100-point rubric.
**Outputs:** `outputs/reports/` (JSON/CSV/MD). **Files:** `src/evaluation/`.

## STEP 18 — Three.js viewer ⬜

**What:** React/Next.js frontend: upload, progress, trajectories, cloud/mesh/
textured views, orbit/pan/zoom, distance/height/area tools with uncertainty
warnings, confidence overlay, exports. **Files:** `frontend/`.

## STEP 19 — FastAPI backend ⬜

**What:** `POST /jobs`, `GET /jobs/{id}[/trajectory|/pointcloud|/mesh|/metrics]`
with the full stage state machine (QUEUED → … → COMPLETED). **Files:** `api/`.

## STEP 20 — Near-real-time optimisation ⬜ (after offline works)

**What:** Lightweight tracking + every-N-frames GPU depth + incremental viewer;
final fusion offline. Report *seconds per video-minute*, never bare
"real-time". No full real-time before the offline pipeline is solid.
