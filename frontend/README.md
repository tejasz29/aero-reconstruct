# Frontend — single-pass-3d viewer (STEP 18, Steps 1–11 scope)

Vite + React + TypeScript + Three.js. Talks to the FastAPI backend (`api/`)
which runs the offline pipeline (Steps 2–11) per job and serves artefacts.

## Run

```bash
# backend (project root)
uvicorn api.main:app --host 127.0.0.1 --port 8000
# frontend (in frontend/)
npm install
npm run dev     # http://127.0.0.1:5173, /api proxied to :8000
npm run build   # production bundle in dist/
```

## What works

- `UploadForm` — `POST /api/jobs` video (+ optional `gps.csv`), queues
  extract-frames → fuse-cloud with dummy depth by default.
- `PipelineProgress` — polls `GET /api/jobs/{id}` + `/logs` every 2s.
- `Viewer3D` — `GET /api/jobs/{id}/trajectory` path (green kept / red
  rejected) + `scene.ply` (STEP 10 raw) or `scene_fused.ply` (STEP 11,
  9-column with normals) via three `PLYLoader`, raw/fused toggle,
  orbit/pan/zoom.
- `ReportsPanel` — calibration / trajectory / GPS / alignment / depth /
  unproject / fusion JSON reports.
- `AccuracyBanner` — relative ≠ absolute; metric-via-scale, NOT absolute
  CRS until STEP 16; occluded = estimated.

## Notes

- Global `outputs/` is used as a fallback when a job has no artefacts yet.
- Large clouds: viewer caps preview quality; use stride thinning
  (`fusion.unproject_stride`) for smooth orbit.
