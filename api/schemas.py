"""STEP 19 — shared job schemas (no heavy deps, no FastAPI import)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Ordered pipeline stages exposed to the frontend (Steps 2-10 only).
STAGES = [
    "extract-frames",    # STEP 2
    "select-keyframes",  # STEP 3
    "calibrate",         # STEP 4
    "reconstruct-poses",  # STEP 5
    "show-trajectory",   # STEP 6
    "convert-gps",       # STEP 7
    "align-trajectory",  # STEP 8
    "predict-depth",     # STEP 9
    "unproject-depth",   # STEP 10
    "fuse-cloud",        # STEP 11
]

JOB_STATUSES = (
    "QUEUED",
    "RUNNING",
    "COMPLETED",
    "COMPLETED_WITH_WARNINGS",
    "FAILED",
    "CANCELLED",
)


@dataclass
class JobStage:
    name: str
    status: str = "pending"  # pending|running|done|failed|skipped
    message: str = ""


@dataclass
class Job:
    id: str
    status: str = "QUEUED"
    created_at: str = ""
    stages: list[JobStage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    params: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def new(job_id: str, params: dict | None = None) -> "Job":
        import datetime

        return Job(
            id=job_id,
            status="QUEUED",
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
            stages=[JobStage(name=s) for s in STAGES],
            params=params or {},
        )
