# single-pass-3d — Full Implementation Steps  session id-ses_f6181e32effeQYcoluE1IoxOJA

Source of truth for build order. A step is **done** only when: implemented,
tested green, demoed on real or synthetic data, README status updated, and
committed + pushed. Never start the next step on a broken tree.

**Progress: STEPS 1–4 done (32 commits + STEP 4 commit(s)) · STEP 5 next · 53/53 tests passing.**

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

## STEP 5 — COLMAP / SfM baseline ⬜

**What:** Camera poses for every keyframe. COLMAP as baseline; custom OpenCV
utilities (SIFT/ORB, matching, RANSAC, essential matrix, PnP) in parallel.
**Inputs:** `keyframes.csv` + intrinsics. **Outputs:**
`outputs/trajectory/camera_poses.csv` (frame ID, timestamp, position,
rotation) + reprojection errors; reject poses above threshold. **Files:**
`src/sfm/`, `src/tracking/` · CLI `reconstruct-poses`. **Tests:** essential-matrix
and PnP recovery on synthetic correspondences.

## STEP 6 — Trajectory visualisation ⬜

**What:** Plot/visualise the estimated camera path to sanity-check SfM before
burning GPU hours. **Outputs:** trajectory plot + `outputs/reports/` figures.
**Files:** `src/sfm/visualize.py` · CLI `show-trajectory`.

## STEP 7 — GPS parsing + metric conversion ⬜

**What:** `timestamp,lat,lon,alt` → metric coordinates. Local ENU for small
scenes, auto UTM for larger ones. Never treat lat/lon as metres.
**Outputs:** `outputs/georef/gps_metric.csv`. **Files:** `src/georef/gps.py`
· CLI `convert-gps`. **Tests:** known-point conversion (distance between two
reference coordinates), ENU origin correctness, UTM zone selection.

## STEP 8 — Visual ↔ GPS alignment ⬜

**What:** Resolve SfM scale ambiguity: `X_global ≈ s·R·X_visual + t` via robust
similarity alignment (RANSAC + least-squares refinement). RTK/PPK path when
available. **Outputs:** aligned trajectory, transformation params, alignment
error. **Files:** `src/georef/align.py` · CLI `align-trajectory`. **Tests:**
recover known s/R/t from synthetic point sets with outliers.

## STEP 9 — Learned depth inference ⬜

**What:** Monocular depth (+ confidence) per keyframe via Depth Anything
family. Depth is NOT metric — constrained later by SfM/GPS scale.
**Outputs:** depth maps + confidence under `outputs/` (or `data/` cache).
**Files:** `src/depth/` · CLI `predict-depth`. **Tests:** output shapes,
finite values, confidence range, determinism smoke test on tiny input.

## STEP 10 — Depth → 3D ⬜

**What:** Unproject (`X=(u-cx)·Z/fx`, …), transform to global frame via poses,
emit colored points. **Files:** `src/fusion/unproject.py`. **Tests:**
unprojection round-trip against known K/Z, pose-transform correctness.

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
