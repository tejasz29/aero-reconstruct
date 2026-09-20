"""STEP 6 — trajectory visualisation.

Reads the STEP 5 ``camera_poses.csv`` and sanity-checks the estimated
flight path before any heavy downstream step burns GPU hours:

* ``read_poses_csv``           — parse the STEP 5 CSV back into poses.
* ``camera_axes``              — world-unit right/up/forward of a pose.
* ``plot_trajectory``          — render a 3D path view (with per-pose
  orientation frustums) and a 2D top-down view; PNGs under ``outputs/reports/``.

Matplotlib is loaded lazily (Agg backend, never a window), so parsing and
unit helpers stay importable without it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.config_loader import get
from src.common.logging_utils import get_logger
from src.sfm.pose import CameraPose

log = get_logger("sp3d.sfm")

# projection name -> (first world axis, second world axis)
TOPDOWN_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
_AXIS_LABEL = ("x", "y", "z")

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_poses_csv(path: str | Path) -> list[CameraPose]:
    """Parse the STEP 5 ``camera_poses.csv`` back into ``CameraPose``s."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    poses: list[CameraPose] = []
    for row in rows:
        kept = row["kept"].strip() in ("1", "True", "true")
        C, R = None, None
        inliers = 0
        err, par = None, None
        if kept:
            C = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])])
            R = np.array([[float(row["r00"]), float(row["r01"]), float(row["r02"])],
                          [float(row["r10"]), float(row["r11"]), float(row["r12"])],
                          [float(row["r20"]), float(row["r21"]), float(row["r22"])]])
            if row.get("inliers"):
                inliers = int(float(row["inliers"]))
            if row.get("mean_reproj_error_px"):
                err = float(row["mean_reproj_error_px"])
            if row.get("median_parallax_px"):
                par = float(row["median_parallax_px"])
        poses.append(CameraPose(
            frame_id=int(row["frame_id"]),
            source_index=int(row.get("source_index") or 0),
            timestamp_s=float(row["timestamp_s"]),
            filename=row["filename"],
            kept=kept,
            R_wc=R, C=C, inliers=inliers,
            mean_reproj_error_px=err, median_parallax_px=par,
            reject_reason=row.get("reject_reason") or ""))
    return poses


def camera_axes(R_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-unit camera axes ``(right, up, forward)`` from a world->camera R.

    OpenCV convention: the camera looks along its +z; rows of ``R_wc`` are
    the camera basis expressed in world coordinates, and image y points down
    (so world *up* is the negated y row).
    """
    R = np.asarray(R_wc, dtype=np.float64)
    right = R[0]
    forward = R[2]
    up = -R[1]
    return right, up, forward


def _topdown_coords(points, projection: str) -> tuple[np.ndarray, np.ndarray]:
    ia, ib = TOPDOWN_AXES[projection]
    pts = np.asarray(points, dtype=np.float64)
    return pts[:, ia], pts[:, ib]


def _frustum_lines(pose: CameraPose, scale: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pyramid conveying the camera's viewpoint for one accepted pose."""
    right, up, forward = camera_axes(pose.R_wc)
    C = np.asarray(pose.C, dtype=np.float64)
    apex = C + scale * forward
    base = C + 0.55 * scale * forward
    w = 0.35 * scale
    corners = [base + dx * right + dy * up
               for dx in (-w, w) for dy in (-w, w)]
    lines = [(apex, c) for c in corners]
    lines += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return lines


def _summary_text(poses: list[CameraPose]) -> str:
    accepted = [p for p in poses if p.kept]
    rejected = [p for p in poses if not p.kept]
    errors = [p.mean_reproj_error_px for p in accepted
              if p.mean_reproj_error_px is not None]
    mean = f"{float(np.mean(errors)):.2f} px" if errors else "-"
    reasons = {}
    for p in rejected:
        reasons[p.reject_reason or "rejected"] = \
            reasons.get(p.reject_reason or "rejected", 0) + 1
    parts = [f"accepted {len(accepted)}/{len(poses)}",
             f"rejected {len(rejected)}", f"mean reproj {mean}"]
    if reasons:
        parts.append("reasons: " + ", ".join(f"{k}={v}" for k, v in
                                             sorted(reasons.items())))
    return "  ·  ".join(parts)


def _equal_aspect(ax, xs, ys, zs, pad_ratio: float = 0.1) -> None:
    ranges = [float(np.max(v) - np.min(v)) for v in (xs, ys, zs)]
    span = max(max(ranges, default=0.0), 1e-9)
    pad = span * pad_ratio
    for data, axis in (xs, "X"), (ys, "Y"), (zs, "Z"):
        lo, hi = float(np.min(data)), float(np.max(data))
        getattr(ax, f"set_{axis.lower()}lim")(lo - pad, hi + pad)
    ax.set_box_aspect((1, 1, 1))


def _plot_3d(plt, fig, kept, draw_frustums: bool, frustum_scale: float) -> None:
    ax = fig.add_subplot(111, projection="3d")
    xs = np.array([p.C[0] for p in kept])
    ys = np.array([p.C[1] for p in kept])
    zs = np.array([p.C[2] for p in kept])
    ax.plot(xs, ys, zs, "-o", color="#1565c0", ms=3, lw=1.2,
            label="camera path")
    origin = kept[0]
    ax.scatter([origin.C[0]], [origin.C[1]], [origin.C[2]], marker="*",
               s=140, color="#ef6c00", label="world origin")
    if draw_frustums:
        for pose in kept:
            for a, b in _frustum_lines(pose, frustum_scale):
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                        color="#90a4ae", lw=0.6, alpha=0.7)
    _equal_aspect(ax, xs, ys, zs)
    ax.set_xlabel("x (relative)")
    ax.set_ylabel("y (relative)")
    ax.set_zlabel("z (relative)")
    ax.legend(loc="best", fontsize=8)


def _plot_topdown(plt, fig, kept, projection: str) -> None:
    ia, ib = TOPDOWN_AXES[projection]
    xs, ys = _topdown_coords([p.C for p in kept], projection)
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, "-o", color="#1565c0", ms=3, lw=1.2, label="camera path")
    origin = kept[0]
    ax.plot([origin.C[ia]], [origin.C[ib]], marker="*", ms=14,
            color="#ef6c00", label="world origin")
    span = max(float(np.max(xs) - np.min(xs)),
               float(np.max(ys) - np.min(ys)), 1e-9)
    arrow_len = 0.15 * span
    for pose in kept:
        _r, _up, forward = camera_axes(pose.R_wc)
        dx, dy = forward[ia] * arrow_len, forward[ib] * arrow_len
        ax.annotate("", xy=(pose.C[ia] + dx, pose.C[ib] + dy),
                    xytext=(pose.C[ia], pose.C[ib]),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="#c62828"))
    ax.set_xlabel(f"{_AXIS_LABEL[ia]} (relative)")
    ax.set_ylabel(f"{_AXIS_LABEL[ib]} (relative)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)


@dataclass
class TrajectoryPlots:
    three_d_path: Path
    topdown_path: Path

    @property
    def paths(self) -> list[Path]:
        return [self.three_d_path, self.topdown_path]


def plot_trajectory(
    poses: list[CameraPose],
    out_dir: str | Path,
    cfg: dict | None = None,
    topdown_projection: str | None = None,
    output_name: str = "trajectory",
) -> TrajectoryPlots:
    """Render 3D + top-down PNGs for ``poses``.

    Accepted poses draw the path, per-pose orientation frustums (3D) and
    forward arrows (top-down); rejected poses contribute to the summary text.
    Raises ``ValueError`` when no accepted pose exists to plot.
    """
    cfg = cfg or {}
    dpi = int(get(cfg, "visualization.dpi", 150))
    figsize = tuple(get(cfg, "visualization.figsize", [10.0, 8.0]))
    projection = topdown_projection or str(
        get(cfg, "visualization.topdown_projection", "xy"))
    if projection not in TOPDOWN_AXES:
        raise ValueError(f"unknown top-down projection: {projection!r} "
                         f"(xy|xz|yz)")
    draw_frustums = bool(get(cfg, "visualization.draw_frustums", True))
    frustum_scale = float(get(cfg, "visualization.frustum_scale", 0.2))

    kept = [p for p in poses if p.kept and p.C is not None and p.R_wc is not None]
    if not kept:
        raise ValueError("no accepted poses to plot — run reconstruct-poses "
                         "first (or check camera_poses.csv has kept=1 rows)")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    three_d_path = out / f"{output_name}_3d.png"
    topdown_path = out / f"{output_name}_topdown.png"

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    title = f"Camera trajectory — {_summary_text(poses)}"
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=11)
    _plot_3d(plt, fig, kept, draw_frustums, frustum_scale)
    fig.savefig(three_d_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"{title}  (top-down {projection})", fontsize=11)
    _plot_topdown(plt, fig, kept, projection)
    fig.savefig(topdown_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    log.info("trajectory figures written: %s, %s", three_d_path, topdown_path)
    return TrajectoryPlots(three_d_path=three_d_path, topdown_path=topdown_path)