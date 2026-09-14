"""Central logging setup for single-pass-3d.

Every pipeline stage must log through this module so that all runs share
one format and one rotating run-log file::

    from src.common.logging_utils import setup_logging, get_logger

    setup_logging()                       # once, at process start
    log = get_logger(__name__)            # in each module
    log.info("extracted %d frames", n)

Configuration comes from ``configs/logging.yaml`` (level, console flag,
file path, rotation). ``setup_logging`` is idempotent: calling it twice
does not duplicate handlers.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Optional

_configured = False

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    console: bool = True,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger once and return it.

    Args:
        level: DEBUG | INFO | WARNING | ERROR (case-insensitive).
        log_file: path of the rotating run log; parent dirs are created.
            ``None`` disables file output.
        console: also log to stderr.
        fmt / datefmt: log record formats.
        max_bytes / backup_count: rotation policy for the file handler.
        force: re-configure even if already configured (useful in tests).
    """
    global _configured
    root = logging.getLogger()
    if _configured and not force:
        return root
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    numeric = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(numeric)
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)

    _configured = True
    return root


def setup_logging_from_config(cfg: dict[str, Any], force: bool = False) -> logging.Logger:
    """Configure logging from the ``logging:`` section of ``configs/logging.yaml``."""
    section = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
    file_cfg = section.get("file", {}) or {}
    log_file = file_cfg.get("path") if file_cfg.get("enabled", False) else None
    return setup_logging(
        level=section.get("level", "INFO"),
        log_file=log_file,
        console=section.get("console", True),
        fmt=section.get("format", DEFAULT_FORMAT),
        datefmt=section.get("datefmt", DEFAULT_DATEFMT),
        max_bytes=int(file_cfg.get("max_bytes", 5 * 1024 * 1024)),
        backup_count=int(file_cfg.get("backup_count", 3)),
        force=force,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Call ``setup_logging`` first at startup."""
    return logging.getLogger(name)
