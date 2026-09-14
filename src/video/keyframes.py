"""STEP 3 — keyframe selection.

Pipeline: read ``timestamps.csv`` (STEP 2) -> score every frame
(``quality.py``) -> quality gates -> greedy dedup/time-gap pass ->
uniform cap at ``max_keep`` -> write ``frame_scores.csv`` + ``keyframes.csv``.

Every rejected frame keeps its ``reject_reason`` (blur | exposure |
features | duplicate | time_gap | max_keep) so STEP 3 decisions stay
auditable. ``min_translation_m`` from config is *not* enforced here —
metric translation is unknown until visual<->GPS alignment (STEP 8).
"""

from __future__ import annotations

import csv
import dataclasses
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from src.common.logging_utils import get_logger
from src.video.frame_extractor import read_timestamps_csv
from src.video.quality import (
    blur_score,
    count_features,
    dhash,
    exposure_mean,
    hamming_distance,
    load_gray,
)

log = get_logger("sp3d.video")

SCORES_CSV = "frame_scores.csv"
KEYFRAMES_CSV = "keyframes.csv"
SCORES_HEADER = ["frame_id", "source_index", "timestamp_s", "filename",
                 "blur", "exposure_mean", "features", "kept", "reject_reason"]
KEYFRAMES_HEADER = ["frame_id", "source_index", "timestamp_s", "filename"]


@dataclass
class FrameScore:
    frame_id: int
    source_index: int
    timestamp_s: float
    filename: str
    blur: float = 0.0
    exposure: float = 0.0
    features: int = 0
    kept: bool = False
    reject_reason: str = ""


@dataclass
class SelectionResult:
    frames_dir: Path
    scores_path: Path
    keyframes_path: Path
    scored: list[FrameScore] = field(default_factory=list)
    kept: list[FrameScore] = field(default_factory=list)

    @property
    def rejected_counts(self) -> dict[str, int]:
        return dict(Counter(s.reject_reason for s in self.scored if not s.kept))


def score_frames(frames_dir: str | Path, detector: str = "orb") -> list[FrameScore]:
    """Score every frame listed in ``timestamps.csv`` (timestamp order)."""
    frames_dir = Path(frames_dir)
    rows = read_timestamps_csv(frames_dir)
    scored: list[FrameScore] = []
    for row in tqdm(rows, desc="scoring", unit="frm"):
        gray = load_gray(frames_dir / row["filename"])
        scored.append(FrameScore(
            frame_id=int(row["frame_id"]),
            source_index=int(row["source_index"]),
            timestamp_s=float(row["timestamp_s"]),
            filename=row["filename"],
            blur=blur_score(gray),
            exposure=exposure_mean(gray),
            features=count_features(gray, detector=detector),
        ))
    log.info("scored %d frames with detector=%s", len(scored), detector)
    return scored


def select_keyframes(
    scored: list[FrameScore],
    hashes: dict[int, int],
    blur_threshold: float,
    exposure_min: float,
    exposure_max: float,
    min_features: int,
    min_time_gap_s: float = 0.0,
    dedup_hamming_threshold: int = 5,
    max_keep: int | None = None,
) -> list[FrameScore]:
    """Apply gates + dedup to scored frames; returns all rows with verdicts.

    Greedy pass in timestamp order: a frame survives only if it passes all
    quality gates, is ``min_time_gap_s`` after the last kept frame, and its
    dHash differs by more than ``dedup_hamming_threshold`` bits from every
    kept frame. Survivors beyond ``max_keep`` are uniform-subsampled.
    Pure function — fully unit-testable.
    """
    ordered = sorted(scored, key=lambda s: s.timestamp_s)
    verdicts = [dataclasses.replace(s) for s in ordered]

    survivors: list[FrameScore] = []
    for row in verdicts:
        if not (exposure_min <= row.exposure <= exposure_max):
            row.reject_reason = "exposure"
        elif row.blur < blur_threshold:
            row.reject_reason = "blur"
        elif row.features < min_features:
            row.reject_reason = "features"
        else:
            survivors.append(row)

    kept: list[FrameScore] = []
    last_time: float | None = None
    for row in survivors:
        if last_time is not None and row.timestamp_s - last_time < min_time_gap_s:
            row.reject_reason = "time_gap"
            continue
        digest = hashes.get(row.frame_id)
        if (digest is not None and kept and dedup_hamming_threshold >= 0
                and any(hamming_distance(digest, hashes.get(k.frame_id, -1))
                        <= dedup_hamming_threshold for k in kept
                        if k.frame_id in hashes)):
            row.reject_reason = "duplicate"
            continue
        row.kept = True
        kept.append(row)
        last_time = row.timestamp_s

    if max_keep is not None and max_keep > 0 and len(kept) > max_keep:
        step = len(kept) / max_keep
        keep_ids = {kept[int(i * step)].frame_id for i in range(max_keep)}
        for row in kept:
            if row.frame_id not in keep_ids:
                row.kept = False
                row.reject_reason = "max_keep"
        kept = [row for row in kept if row.kept]

    log.info("keyframes: %d kept / %d scored", len(kept), len(verdicts))
    return verdicts


def write_scores_csv(path: str | Path, scored: list[FrameScore]) -> Path:
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(SCORES_HEADER)
        for s in scored:
            writer.writerow([s.frame_id, s.source_index, f"{s.timestamp_s:.6f}",
                             s.filename, f"{s.blur:.3f}", f"{s.exposure:.3f}",
                             s.features, int(s.kept), s.reject_reason])
    return path


def write_keyframes_csv(path: str | Path, kept: list[FrameScore]) -> Path:
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(KEYFRAMES_HEADER)
        for s in kept:
            writer.writerow([s.frame_id, s.source_index,
                             f"{s.timestamp_s:.6f}", s.filename])
    return path


def run_selection(
    frames_dir: str | Path,
    blur_threshold: float,
    exposure_min: float,
    exposure_max: float,
    min_features: int,
    min_time_gap_s: float = 0.0,
    dedup_hamming_threshold: int = 5,
    max_keep: int | None = None,
    detector: str = "orb",
) -> SelectionResult:
    """Full STEP 3 run: score -> select -> write both CSVs."""
    frames_dir = Path(frames_dir)
    scored = score_frames(frames_dir, detector=detector)
    hashes = {s.frame_id: dhash(load_gray(frames_dir / s.filename)) for s in scored}
    verdicts = select_keyframes(
        scored, hashes, blur_threshold, exposure_min, exposure_max,
        min_features, min_time_gap_s, dedup_hamming_threshold, max_keep)
    kept = [s for s in verdicts if s.kept]
    result = SelectionResult(
        frames_dir=frames_dir,
        scores_path=write_scores_csv(frames_dir / SCORES_CSV, verdicts),
        keyframes_path=write_keyframes_csv(frames_dir / KEYFRAMES_CSV, kept),
        scored=verdicts,
        kept=kept,
    )
    log.info("selection: %d kept, rejected=%s", len(kept), result.rejected_counts)
    return result
