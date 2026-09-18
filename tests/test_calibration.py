"""STEP 4 — camera intrinsics model + board calibration."""

import json

import cv2
import numpy as np
import pytest
import yaml

from src.calibration.charuco import calibrate_charuco, charuco_board
from src.calibration.checkerboard import (
    calibrate_checkerboard,
    detect_checkerboard,
)
from src.calibration.intrinsics import (
    Intrinsics,
    load_intrinsics,
    make_intrinsics,
    project,
    save_intrinsics,
    undistort_image,
    unproject_pixel,
)
from src.calibration.runner import resolve_calibration

GT = dict(fx=700.0, fy=700.0, cx=160.0, cy=120.0, width=320, height=240)


def gt_intrinsics(**overrides):
    params = dict(GT)
    params.update(overrides)
    del params["width"], params["height"]
    return make_intrinsics(width=GT["width"], height=GT["height"], **params)


# --- intrinsics model ---

def test_camera_matrix_layout():
    cam = gt_intrinsics()
    K = cam.camera_matrix()
    assert K.shape == (3, 3)
    assert K[0, 0] == GT["fx"] and K[1, 1] == GT["fy"]
    assert K[0, 2] == GT["cx"] and K[1, 2] == GT["cy"]
    assert K[0, 1] == 0.0 and K[2].tolist() == [0.0, 0.0, 1.0]


def test_projection_unprojection_round_trip():
    cam = gt_intrinsics()
    pts = np.array([[0.2, -0.1, 5.0], [-1.5, 2.0, 8.0], [0.0, 0.0, 3.0]])
    pixels = project(pts, cam)
    assert pixels.shape == (3, 2)
    for i, depth in enumerate(pts[:, 2]):
        recovered = unproject_pixel(*pixels[i], float(depth), cam)
        np.testing.assert_allclose(recovered, pts[i], atol=1e-6)


def test_unprojection_requires_positive_depth():
    with pytest.raises(ValueError):
        unproject_pixel(5, 5, -1.0, gt_intrinsics())


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "camera.yaml"
    cam = Intrinsics(fx=GT["fx"], fy=GT["fy"], cx=GT["cx"], cy=GT["cy"],
                     width=GT["width"], height=GT["height"],
                     source="checkerboard", reprojection_error_px=0.42,
                     distortion=(0.1, -0.2, 0.001, 0.0, 0.3))
    save_intrinsics(cam, path)
    loaded = load_intrinsics(path)
    assert cam == loaded
    assert loaded.distortion == pytest.approx((0.1, -0.2, 0.001, 0.0, 0.3))


def test_validate_rejects_nonsense():
    with pytest.raises(ValueError):
        Intrinsics(fx=-5, fy=700, cx=160, cy=120, width=320, height=240).validate()
    with pytest.raises(ValueError):
        Intrinsics(fx=700, fy=700, cx=160, cy=120, width=0, height=240).validate()
    with pytest.raises(ValueError):
        Intrinsics(fx=700, fy=700, cx=160, cy=120, width=320, height=240,
                   distortion=(0.1,)).validate()


def test_undistort_preserves_shape():
    cam = gt_intrinsics(distortion=(0.05, -0.02, 0.0, 0.0, 0.0))
    img = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    out = undistort_image(img, cam)
    assert out.shape == img.shape


# --- synthetic checkerboard (true-camera rendering) ---

def _draw_checkerboard(sq_px=24, cols=9, rows=6, margin=0):
    """Edge-to-edge board image: (cols+1) x (rows+1) squares of ``sq_px``."""
    board_w = (cols + 1) * sq_px + 2 * margin
    board_h = (rows + 1) * sq_px + 2 * margin
    img = np.full((board_h, board_w), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                x0, y0 = margin + c * sq_px, margin + r * sq_px
                img[y0:y0 + sq_px, x0:x0 + sq_px] = 0
    return img


def _render_board_views(K, width, height, cols=9, rows=6, sq_size=0.025,
                        n_views=8, seed=5, sq_px=28):
    """Render the flat board through the KNOWN camera K via its plane homography.

    ``H = K @ [mpp*R[:,0], mpp*R[:,1], t]`` maps texture pixels to the image;
    meters-per-pixel is exact because the edge-to-edge board is
    ``(cols+1)*sq_size`` m wide by ``(rows+1)*sq_size`` m tall.
    """
    board = _draw_checkerboard(sq_px=sq_px, cols=cols, rows=rows, margin=0)
    bh, bw = board.shape[:2]
    mpp = sq_size / sq_px
    rng = np.random.default_rng(seed)
    views = []
    for _ in range(n_views):
        for _attempt in range(120):
            t = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25),
                          rng.uniform(0.7, 1.2)])
            rvec = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35),
                             rng.uniform(-0.35, 0.35)])
            R, _ = cv2.Rodrigues(rvec)
            H = K @ np.hstack([R[:, 0:1] * mpp, R[:, 1:2] * mpp, t.reshape(3, 1)])
            corners = np.float32([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]])
            proj = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
            margin = 20
            fits = (proj[:, 0].min() > margin and proj[:, 0].max() < width - margin
                    and proj[:, 1].min() > margin and proj[:, 1].max() < height - margin)
            w_span = proj[:, 0].max() - proj[:, 0].min()
            h_span = proj[:, 1].max() - proj[:, 1].min()
            if fits and min(w_span, h_span) >= 130:
                break
        else:
            raise RuntimeError("could not place board inside the image")
        views.append(cv2.warpPerspective(board, H, (width, height), borderValue=255))
    return views


def test_checkerboard_calibrates_to_ground_truth(tmp_path):
    K = np.array([[700.0, 0, 160.0], [0, 700.0, 120.0], [0, 0, 1]])
    views = _render_board_views(K, 320, 240)
    images = []
    for i, view in enumerate(views):
        path = tmp_path / f"view_{i:02d}.jpg"
        cv2.imwrite(str(path), view)
        images.append(path)
    result = calibrate_checkerboard(images, (9, 6), 0.025, image_size=(320, 240))
    assert len(result.used_images) == len(views)
    assert result.rms_px < 0.5
    assert result.intrinsics.fx == pytest.approx(700.0, abs=20)
    assert result.intrinsics.fy == pytest.approx(700.0, abs=20)
    assert result.intrinsics.cx == pytest.approx(160.0, abs=10)
    assert result.intrinsics.cy == pytest.approx(120.0, abs=10)


def test_detect_checkerboard_positive_and_negative(tmp_path):
    board = _draw_checkerboard()
    assert detect_checkerboard(board, (9, 6)) is not None
    blank = np.full((100, 100), 128, np.uint8)
    assert detect_checkerboard(blank, (9, 6)) is None


def test_checkerboard_needs_min_views(tmp_path):
    K = np.array([[700.0, 0, 160.0], [0, 700.0, 120.0], [0, 0, 1]])
    views = _render_board_views(K, 320, 240, n_views=2)
    images = []
    for i, view in enumerate(views):
        path = tmp_path / f"v{i}.jpg"
        cv2.imwrite(str(path), view)
        images.append(path)
    with pytest.raises(ValueError):
        calibrate_checkerboard(images, (9, 6), 0.025, image_size=(320, 240))


# --- synthetic Charuco (homography rendering of the real board texture) ---

def _render_charuco_views(K, width, height, rows=5, cols=7, square=0.03,
                          marker=0.02, n_views=8, seed=11):
    """Perspective-render the printed Charuco board through the known K.

    The planar homography ``H = K @ [s*R[:,0], s*R[:,1], t]`` maps the
    printed texture (pixels) to the view; ``s`` converts texture pixels to
    metres (full board width spans ``cols*square`` m).
    """
    from src.calibration.charuco import generate_charuco_view

    board = charuco_board(rows=rows, cols=cols, square_length_m=square,
                          marker_length_m=marker)
    texture = generate_charuco_view(board, 1400, 1000)
    bh, bw = texture.shape[:2]
    m_per_px_x = (cols * square) / bw
    m_per_px_y = (rows * square) / bh
    rng = np.random.default_rng(seed)
    views = []
    for _ in range(n_views):
        for _attempt in range(120):
            t = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08),
                          rng.uniform(0.7, 0.95)])
            rvec = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25),
                             rng.uniform(-0.25, 0.25)])
            R, _ = cv2.Rodrigues(rvec)
            H = K @ np.hstack([R[:, 0:1] * m_per_px_x, R[:, 1:2] * m_per_px_y,
                               t.reshape(3, 1)])
            corners = np.float32([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]])
            proj = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
            fits = (proj[:, 0].min() > 8 and proj[:, 0].max() < width - 8
                    and proj[:, 1].min() > 8 and proj[:, 1].max() < height - 8)
            w_span = proj[:, 0].max() - proj[:, 0].min()
            h_span = proj[:, 1].max() - proj[:, 1].min()
            # Board close enough that markers occupy >= ~18 px and corners
            # interpolate cleanly.
            if fits and min(w_span, h_span) >= 150:
                break
        else:
            raise RuntimeError("could not place charuco board inside the image")
        img = cv2.warpPerspective(texture, H, (width, height), borderValue=255)
        views.append(img)
    return views


def test_charuco_calibrates_to_ground_truth(tmp_path):
    K = np.array([[700.0, 0, 160.0], [0, 700.0, 120.0], [0, 0, 1]])
    views = _render_charuco_views(K, 320, 240)
    images = []
    for i, view in enumerate(views):
        path = tmp_path / f"ch_{i:02d}.jpg"
        cv2.imwrite(str(path), view)
        images.append(path)
    result = calibrate_charuco(images, 5, 7, 0.04, 0.02, image_size=(320, 240))
    assert len(result.used_images) >= 6
    assert result.rms_px < 2.0
    assert result.intrinsics.fx == pytest.approx(700.0, rel=0.2)
    assert result.intrinsics.fy == pytest.approx(700.0, rel=0.2)


# --- runner / config-driven ---

def test_resolve_provided_writes_model(tmp_path):
    provided = tmp_path / "camera_provided.yaml"
    with open(provided, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"camera": {"source": "provided",
                                   "image_width": GT["width"],
                                   "image_height": GT["height"],
                                   "fx": GT["fx"], "fy": GT["fy"],
                                   "cx": GT["cx"], "cy": GT["cy"],
                                   "distortion": [0.0] * 5,
                                   "reprojection_error_px": None,
                                   "calibrated_on": None}}, fh)
    out = tmp_path / "camera.yaml"
    cfg = {"calibration": {"source": "provided", "provided_file": str(provided),
                           "file": str(out)},
           "paths": {"reports": str(tmp_path / "reports")}}
    _, intrinsics = resolve_calibration(cfg, output_path=out)
    assert intrinsics.source == "provided"
    assert intrinsics.fx == GT["fx"]
    assert load_intrinsics(out).fx == GT["fx"]


def test_resolve_provided_incomplete_raises(tmp_path):
    provided = tmp_path / "camera_provided.yaml"
    with open(provided, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"camera": {"source": "provided", "fx": 700}}, fh)
    with pytest.raises(ValueError):
        resolve_calibration({"calibration": {"source": "provided",
                                             "provided_file": str(provided)},
                             "paths": {"reports": str(tmp_path / "reports")}},
                            output_path=tmp_path / "camera.yaml")


def test_resolve_checkerboard_end_to_end(tmp_path):
    K = np.array([[700.0, 0, 160.0], [0, 700.0, 120.0], [0, 0, 1]])
    images = tmp_path / "images"
    images.mkdir()
    for i, view in enumerate(_render_board_views(K, 320, 240)):
        cv2.imwrite(str(images / f"view_{i:02d}.jpg"), view)
    out = tmp_path / "camera.yaml"
    cfg = {"calibration": {"source": "checkerboard",
                           "checkerboard": {"rows": 6, "cols": 9,
                                            "square_size_m": 0.025},
                           "validation": {"max_reprojection_error_px": 5.0},
                           "file": str(out)},
           "paths": {"reports": str(tmp_path / "reports")}}
    result, intrinsics = resolve_calibration(cfg, images_dir=images, output_path=out)
    assert intrinsics.source == "checkerboard"
    assert out.is_file()
    report = json.loads((tmp_path / "reports" / "calibration_report.json").read_text())
    assert report["within_threshold"] is True
    assert report["used_views"] == len(result.used_images) == 8


def test_unknown_source_raises(tmp_path):
    cfg = {"calibration": {"source": "bogus"},
           "paths": {"reports": str(tmp_path / "reports")}}
    with pytest.raises(ValueError):
        resolve_calibration(cfg, output_path=tmp_path / "camera.yaml")