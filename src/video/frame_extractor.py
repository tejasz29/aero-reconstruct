"""STEP 2 — frame extraction.

Thins the source video to at most ``target_fps`` frames/second, caps the
total at ``max_frames`` (uniform subsample), downscales frames wider than
``max_width``, and writes ``timestamps.csv`` next to the images.

Layout of ``out_dir`` after a run::

    frame_000000.jpg
    frame_000001.jpg
    ...
    timestamps.csv   # frame_id,source_index,timestamp_s,filename,width,height

Quality scoring and keyframe *selection* are STEP 3 — everything sampled
here is kept.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from tqdm import tqdm

from src.common.logging_utils import get_logger
from src.video.video_info import VideoInfo, frame_timestamp, probe_video

log = get_logger("sp3d.video")

TIMESTAMPS_CSV = "timestamps.csv"
CSV_HEADER = ["frame_id", "source_index", "timestamp_s", "filename", "width", "height"]


@dataclass(frozen=True)
class ExtractedFrame:
    frame_id: int
    source_index: int
    timestamp_s: float
    filename: str
    width: int
    height: int


@dataclass
class ExtractionResult:
    info: VideoInfo
    out_dir: Path
    csv_path: Path
    frames: list[ExtractedFrame] = field(default_factory=list)


def compute_sample_indices(
    n_frames: int,
    src_fps: float,
    target_fps: float | None,
    max_frames: int | None,
) -> list[int]:
    """Evenly spaced 0-based source indices to keep.

    * stride = round(src_fps / target_fps); ``target_fps >= src_fps``
      (or unset/non-positive) keeps every frame;
    * if more than ``max_frames`` survive, uniform-subsample down to it.
    Pure function — no I/O, fully unit-testable.
    """
    if n_frames <= 0:
        return []
    if target_fps is None or target_fps <= 0 or target_fps >= src_fps:
        indices = list(range(n_frames))
    else:
        stride = max(1, int(round(src_fps / target_fps)))
        indices = list(range(0, n_frames, stride))
    if max_frames is not None and max_frames > 0 and len(indices) > max_frames:
        step = len(indices) / max_frames
        indices = [indices[int(i * step)] for i in range(max_frames)]
    return indices


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    target_fps: float = 2.0,
    max_frames: int = 2000,
    max_width: int = 1920,
    output_format: str = "jpg",
    jpeg_quality: int = 95,
) -> ExtractionResult:
    """Extract frames from ``video_path`` into ``out_dir`` (created if needed)."""
    info = probe_video(video_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ext = "png" if str(output_format).lower() == "png" else "jpg"
    indices = compute_sample_indices(info.frame_count, info.fps, target_fps, max_frames)
    wanted = set(indices)
    if not wanted:
        log.warning("no frames selected (empty/short video?)")

    scale = min(1.0, max_width / info.width) if info.width > 0 else 1.0
    new_w, new_h = int(info.width * scale), int(info.height * scale)
    if scale < 1.0:
        log.info("downscaling frames %dx%d -> %dx%d", info.width, info.height, new_w, new_h)

    rows: list[ExtractedFrame] = []
    cap = cv2.VideoCapture(str(info.path))
    try:
        for src_idx in tqdm(range(info.frame_count), desc="extracting", unit="frm"):
            ok, frame = cap.read()
            if not ok:
                log.warning("decoder stopped at source frame %d/%d",
                            src_idx, info.frame_count)
                break
            if src_idx not in wanted:
                continue
            if scale < 1.0:
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame_id = len(rows)
            filename = f"frame_{frame_id:06d}.{ext}"
            params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)] if ext == "jpg" else []
            if not cv2.imwrite(str(out / filename), frame, params):
                raise IOError(f"failed to write {out / filename}")
            rows.append(ExtractedFrame(
                frame_id=frame_id,
                source_index=src_idx,
                timestamp_s=frame_timestamp(src_idx, info.fps),
                filename=filename,
                width=frame.shape[1],
                height=frame.shape[0],
            ))
    finally:
        cap.release()

    csv_path = out / TIMESTAMPS_CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow([row.frame_id, row.source_index,
                             f"{row.timestamp_s:.6f}", row.filename,
                             row.width, row.height])

    log.info("extracted %d/%d frames -> %s (+ %s)",
             len(rows), info.frame_count, out, TIMESTAMPS_CSV)
    return ExtractionResult(info=info, out_dir=out, csv_path=csv_path, frames=rows)
