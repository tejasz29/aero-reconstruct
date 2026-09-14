"""STEP 1 — logging writes to console + rotating file, exactly once."""

import logging

from src.common import logging_utils
from src.common.logging_utils import get_logger, setup_logging, setup_logging_from_config


def test_setup_logging_creates_log_file(tmp_path):
    log_file = tmp_path / "logs" / "run.log"  # parent must be auto-created
    setup_logging(level="DEBUG", log_file=log_file, console=False, force=True)
    get_logger("test.stage").info("hello-log-file")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert log_file.is_file()
    assert "hello-log-file" in log_file.read_text(encoding="utf-8")


def test_setup_logging_is_idempotent():
    setup_logging(level="INFO", console=True, force=True)
    before = len(logging.getLogger().handlers)
    setup_logging(level="INFO", console=True)  # second call: no new handlers
    assert len(logging.getLogger().handlers) == before


def test_logging_yaml_config_drives_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # relative log path resolves under tmp dir
    cfg = {
        "logging": {
            "level": "WARNING",
            "console": False,
            "format": "%(message)s",
            "file": {"enabled": True, "path": "out/run.log",
                     "max_bytes": 1024, "backup_count": 1},
        }
    }
    setup_logging_from_config(cfg, force=True)
    assert logging.getLogger().level == logging.WARNING
    get_logger("test.yaml").warning("yaml-driven")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "yaml-driven" in (tmp_path / "out" / "run.log").read_text(encoding="utf-8")
    # restore quiet console logging for the rest of the session
    logging_utils._configured = False
    setup_logging(level="WARNING", console=False, force=True)
