"""STEP 11 — voxel fusion unit tests (numpy baseline, no open3d needed)."""

import numpy as np

from src.fusion.fuse import voxel_downsample, voxel_keys


def test_voxel_keys_quantize():
    pts = np.array([[0.01, 0.0, 0.0], [0.15, 0.0, 0.0]])
    keys = voxel_keys(pts, 0.10)
    assert keys[0, 0] == 0 and keys[1, 0] == 1


def test_voxel_keys_reject_bad_size():
    import pytest

    with pytest.raises(ValueError):
        voxel_keys(np.ones((2, 3)), 0.0)


def test_voxel_collapses_duplicates():
    pts = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cols = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    p, _c, _f = voxel_downsample(pts, cols, voxel_size=0.1)
    assert len(p) == 2  # first two share a voxel


def test_weighted_average_prefers_confident_point():
    pts = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    cols = np.array([[0, 0, 0], [200, 200, 200]], dtype=np.uint8)
    conf = np.array([0.05, 0.95])
    p, c, _f = voxel_downsample(pts, cols, conf, voxel_size=0.5)
    assert len(p) == 1
    assert p[0, 0] > 0.03  # pulled toward the confident point
    assert c[0, 0] > 150


def test_downsample_deterministic():
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 1, (50, 3))
    cols = (rng.uniform(0, 255, (50, 3))).astype(np.uint8)
    a = voxel_downsample(pts, cols, voxel_size=0.2)
    b = voxel_downsample(pts[::-1], cols[::-1], voxel_size=0.2)
    assert np.allclose(np.sort(a[0].ravel()), np.sort(b[0].ravel()))


def test_downsample_rejects_empty_and_mismatch():
    import pytest

    with pytest.raises(ValueError):
        voxel_downsample(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        voxel_downsample(np.ones((2, 3)),
                         np.ones((3, 3), dtype=np.uint8))


def test_normals_reject_empty():
    import pytest

    from src.fusion.fuse import estimate_normals_pca

    with pytest.raises(ValueError):
        estimate_normals_pca(np.zeros((0, 3)))


def test_pca_normals_planar_cloud():
    from src.fusion.fuse import estimate_normals_pca, orient_normals

    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-1, 1, 40), rng.uniform(-1, 1, 40),
                           np.zeros(40)])
    nrm = orient_normals(pts, estimate_normals_pca(pts, k=6))
    assert nrm.shape == (40, 3)
    assert float(np.abs(nrm[:, 2]).mean()) > 0.99
    assert np.all(np.isfinite(nrm))


def test_fuse_clouds_stats_and_normals():
    from src.fusion.fuse import fuse_clouds

    rng = np.random.default_rng(3)
    pts = np.vstack([rng.uniform(0, 1, (30, 3)), rng.uniform(0, 1, (30, 3))])
    cols = (rng.uniform(0, 255, (60, 3))).astype(np.uint8)
    out = fuse_clouds(pts, cols, voxel_size=0.25)
    assert out["n_in"] == 60
    assert out["n_out"] <= 60
    assert out["normals"] is not None
    assert out["normals_backend"] in ("pca", "open3d")


def test_ply_normals_roundtrip(tmp_path):
    from src.fusion.io import (load_ply, load_ply_with_normals, save_ply,
                               validate_cloud)

    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cols = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    nrm = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    p = save_ply(pts, cols, tmp_path / "n.ply", normals=nrm)
    back_pts, back_cols = load_ply(p)  # backward compat: ignores normals
    assert np.allclose(back_pts, pts, atol=1e-6)
    _, _, back_nrm = load_ply_with_normals(p)
    assert back_nrm is not None and np.allclose(back_nrm, nrm, atol=1e-6)
    validate_cloud(back_pts, back_cols, back_nrm)


def test_run_fusion_e2e_synthetic(tmp_path):
    from src.common.config_loader import load_config
    from src.fusion.io import save_ply, write_cloud_index
    from src.fusion.runner import run_fusion

    rng = np.random.default_rng(11)
    frames = tmp_path / "frames"
    frames.mkdir()
    out = tmp_path / "cloud"
    out.mkdir()
    rows = []
    for i in range(2):
        pts = rng.uniform(0, 1, (40, 3)) + np.array([i * 0.05, 0.0, 0.0])
        cols = (rng.uniform(0, 255, (40, 3))).astype(np.uint8)
        ply = out / f"frame_{i}.ply"
        save_ply(pts, cols, ply)
        rows.append({"frame_id": i, "filename": f"f{i}.jpg",
                     "n_points": 40, "ply_path": str(ply),
                     "mean_confidence": 0.8})
    index = write_cloud_index(rows, out / "cloud_index.csv")
    cfg = load_config()
    res = run_fusion(cfg, cloud_index=index, output_dir=out, voxel_size=0.2)
    assert res["n_in"] == 80
    assert res["n_points"] <= 80
    assert (out / "scene_fused.ply").is_file()


def test_fusion_policy_and_parser():
    from src.common.config_loader import load_config
    from src.fusion.runner import resolve_fusion_policy

    import pytest

    cfg = load_config()
    pol = resolve_fusion_policy(cfg, voxel_size=0.25)
    assert pol.voxel_size_m == 0.25
    with pytest.raises(ValueError):
        resolve_fusion_policy(cfg, voxel_size=0.0)
    cv2 = pytest.importorskip("cv2", reason="full pipeline deps missing")
    assert cv2 is not None
    from src.cli import build_parser

    args = build_parser().parse_args(["fuse-cloud"])
    assert args.command == "fuse-cloud"


def test_run_fusion_rejects_empty_index(tmp_path):
    import pytest

    from src.common.config_loader import load_config
    from src.fusion.runner import run_fusion

    empty = tmp_path / "cloud_index.csv"
    empty.write_text("frame_id,filename,n_points,ply_path,mean_confidence\n")
    with pytest.raises(ValueError):
        run_fusion(load_config(), cloud_index=empty, output_dir=tmp_path)
