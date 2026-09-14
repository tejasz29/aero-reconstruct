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
