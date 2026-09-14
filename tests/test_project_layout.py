"""STEP 1 — the on-disk layout must match the documented structure."""

from src.common.paths import project_paths

# Every directory the pipeline stages will read from / write to.
EXPECTED = [
    "raw_videos", "frames", "gps", "imu", "reference",
    "calibration", "models_depth", "models_segmentation",
    "outputs", "pointcloud", "mesh", "textures", "reports",
    "trajectory", "georef", "configs",
]


def test_all_layout_directories_exist():
    paths = project_paths()
    missing = paths.missing_dirs()
    assert not missing, f"missing project directories: {missing}"


def test_expected_directory_count():
    paths = project_paths()
    assert len(EXPECTED) == 16
    for name in EXPECTED:
        assert getattr(paths, name).is_dir(), f"{name} is not a directory"
