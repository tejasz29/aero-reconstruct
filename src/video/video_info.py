"""STEP 2 — video probing.

Reads container metadata (FPS, resolution, frame count, codec) without
decoding pixel data, and converts source frame indices to seconds.

Timestamp model: ``t = source_index / fps``. This is exact for
constant-frame-rate (CFR) drone footage. For variable-frame-rate (VFR)
sources it is an approximation — STEP 3+ consumers that need
frame-accurate sync (e.g. GPS interpolation) must verify against a
CFR-transcoded input (``ffmpeg -vsync cfr``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from src.common.logging_utils import get_logger

log = get_logger("sp3d.video")


@dataclass(frozen=True)
class VideoInfo:
    """Container metadata for one input video."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str


def frame_timestamp(source_index: int, fps: float) -> float:
    """Timestamp in seconds of a 0-based source frame index at ``fps``."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if source_index < 0:
        raise ValueError(f"source_index must be >= 0, got {source_index}")
    return source_index / fps


def _fourcc_to_str(fourcc: int) -> str:
    return "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))


def probe_video(path: str | Path) -> VideoInfo:
    """Open ``path`` with OpenCV and return its metadata.

    Falls back to decoding-and-counting when the container reports no
    frame count (some MP4/MOV files); raises FileNotFoundError /
    ValueError for missing or unreadable inputs.
    """
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video (unsupported codec?): {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = _fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))

        if count <= 0:
            log.warning("container reports no frame count — counting by decoding")
            count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
        if fps <= 0:
            raise ValueError(f"could not determine FPS for {video_path} (VFR source?)")
    finally:
        cap.release()

    info = VideoInfo(
        path=video_path,
        width=width,
        height=height,
        fps=fps,
        frame_count=count,
        duration_s=count / fps,
        codec=codec,
    )
    log.info(
        "probed %s: %dx%d @ %.2f fps, %d frames (%.1fs), codec=%s",
        video_path.name, width, height, fps, count, info.duration_s, codec,
    )
    return info
