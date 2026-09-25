"""STEP 1 — config loading: defaults resolve, overrides merge, get() is safe."""

import pytest
import yaml

from src.common.config_loader import get, load_config, load_yaml
from src.common.paths import PROJECT_ROOT

# One section per future pipeline stage — config must exist before code.
REQUIRED_SECTIONS = [
    "project", "paths", "video", "calibration", "sfm", "gps",
    "alignment", "depth", "segmentation", "fusion", "mesh",
    "texture", "georef", "evaluation", "api", "runtime",
]


def test_default_config_loads_with_all_sections():
    cfg = load_config()
    for section in REQUIRED_SECTIONS:
        assert section in cfg, f"missing config section: {section}"


def test_config_values_have_expected_types():
    cfg = load_config()
    assert isinstance(get(cfg, "video.target_fps"), (int, float))
    assert get(cfg, "video.quality.blur_threshold") > 0
    assert get(cfg, "sfm.backend") == "colmap"
    assert isinstance(get(cfg, "segmentation.mask_classes"), list)


def test_run_override_deep_merges(tmp_path):
    override = tmp_path / "run.yaml"
    override.write_text(
        yaml.safe_dump({"video": {"target_fps": 5.0}, "project": {"seed": 7}}),
        encoding="utf-8",
    )
    cfg = load_config(override)
    assert get(cfg, "video.target_fps") == 5.0
    # sibling keys from the base config must survive the merge
    assert get(cfg, "video.max_frames") == load_config()["video"]["max_frames"]
    assert get(cfg, "project.seed") == 7


def test_get_returns_default_for_missing_keys():
    cfg = load_config()
    assert get(cfg, "nope.not.there", "fallback") == "fallback"
    assert get(cfg, "video.nope", None) is None


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "does-not-exist.yaml")


def test_camera_stub_exists_and_is_unfilled():
    cam = load_yaml(PROJECT_ROOT / "calibration" / "camera.yaml")
    assert cam["camera"]["source"] == "placeholder"
    assert cam["camera"]["fx"] is None  # filled in STEP 4
