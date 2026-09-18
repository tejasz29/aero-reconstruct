"""STEP 4 — camera intrinsics model + board calibration."""

import cv2
import numpy as np
import pytest

from src.calibration.intrinsics import (
    Intrinsics,
    load_intrinsics,
    make_intrinsics,
    project,
    save_intrinsics,
    undistort_image,
    unproject_pixel,
)

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