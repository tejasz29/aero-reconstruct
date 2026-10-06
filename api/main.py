"""STEP 19 — FastAPI application factory (Steps 1-10 scope).

Commit 1 wires only liveness + router placeholders; jobs/artifacts
routers land in commit 2. Heavy pipeline imports stay inside
``api.pipeline`` so ``import api.main`` never needs torch/open3d.
"""

from __future__ import annotations

PIPELINE_STEPS_DONE = list(range(1, 11))
PIPELINE_NEXT = 11


def pipeline_progress() -> dict:
    return {
        "steps_done": PIPELINE_STEPS_DONE,
        "steps_total": 20,
        "next_step": PIPELINE_NEXT,
        "note": "Steps 1-10 done: video -> keyframes -> calibration -> "
        "poses -> trajectory viz -> GPS -> alignment -> depth -> 3D clouds.",
    }


def create_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="single-pass-3d", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "service": "single-pass-3d", **pipeline_progress()}

    # Jobs + artifacts routers are registered when available (commit 2).
    try:  # pragma: no cover - import-time wiring only
        from .artifacts import router as artifacts_router  # noqa: F401
        from .jobs import router as jobs_router  # noqa: F401

        app.include_router(jobs_router)
        app.include_router(artifacts_router)
    except ImportError:
        pass

    return app


app = None
try:  # allow `uvicorn api.main:app` while keeping plain `import api.main` light
    app = create_app()
except ImportError:  # FastAPI not installed in minimal envs
    app = None
