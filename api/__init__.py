"""STEP 19 — FastAPI backend (offline MVP, Steps 1-10 scope).

Endpoints (full set lands across commits):
  GET  /api/health            -> liveness + pipeline progress (steps 1-10 done)
  POST /api/jobs              -> upload video (+gps) and queue Steps 2-10
  GET  /api/jobs[/{id}...]    -> status, trajectory, pointcloud, reports
"""

__all__ = []
