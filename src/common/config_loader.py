"""YAML configuration loading for single-pass-3d.

Single entry point::

    from src.common.config_loader import load_config, get

    cfg = load_config()                       # configs/default.yaml
    cfg = load_config("configs/my_run.yaml")  # custom run file
    blur = get(cfg, "video.quality.blur_threshold", 100.0)

Rules:
* ``configs/default.yaml`` is always the base (keeps every stage's keys).
* An optional run file is deep-merged on top of the base.
* No stage may read a YAML file directly — it goes through here so that
  every run logs exactly one resolved configuration.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from src.common.paths import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
ENV_CONFIG_VAR = "SP3D_CONFIG"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return it as a dict (empty file -> {})."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (new dict, inputs untouched)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(run_config: str | Path | None = None) -> dict[str, Any]:
    """Load the resolved pipeline configuration.

    Resolution order: ``configs/default.yaml`` < ``run_config`` file
    < ``SP3D_CONFIG`` env var (if set and no explicit file given).
    """
    base = load_yaml(DEFAULT_CONFIG_PATH)
    if run_config is None:
        env_path = os.environ.get(ENV_CONFIG_VAR)
        run_config = env_path if env_path else None
    if run_config is not None:
        override = load_yaml(run_config)
        base = _deep_merge(base, override)
    return base


def get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch ``a.b.c`` from a nested config dict, returning ``default`` if absent."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
