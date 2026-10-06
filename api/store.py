"""STEP 19 — tiny file-based job store (no DB for the offline MVP)."""

from __future__ import annotations

import json
from pathlib import Path

from src.common.paths import PROJECT_ROOT

from .schemas import Job, JobStage


def jobs_root() -> Path:
    root = PROJECT_ROOT / "outputs" / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def job_dir(job_id: str) -> Path:
    d = jobs_root() / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def job_json_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def save_job(job: Job) -> Path:
    p = job_json_path(job.id)
    p.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
    return p


def load_job(job_id: str) -> Job:
    p = jobs_root() / job_id / "job.json"
    if not p.is_file():
        raise FileNotFoundError(f"job not found: {job_id}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    stages = [JobStage(**s) for s in raw.get("stages", [])]
    return Job(
        id=raw.get("id", job_id),
        status=raw.get("status", "QUEUED"),
        created_at=raw.get("created_at", ""),
        stages=stages,
        warnings=raw.get("warnings", []),
        error=raw.get("error", ""),
        params=raw.get("params", {}),
        outputs=raw.get("outputs", {}),
    )


def list_jobs() -> list[Job]:
    root = jobs_root()
    jobs: list[Job] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "job.json").is_file():
            try:
                jobs.append(load_job(child.name))
            except (OSError, ValueError):
                continue
    return sorted(jobs, key=lambda j: j.created_at, reverse=True)


def append_log(job_id: str, line: str) -> Path:
    p = job_dir(job_id) / "run.log"
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")
    return p
