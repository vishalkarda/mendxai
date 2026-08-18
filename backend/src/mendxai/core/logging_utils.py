"""Process-wide logging: one timestamped log file per run, shared by every
module via a singleton root-logger configuration.

Usage:
    from mendxai.core.logging_utils import get_logger
    logger = get_logger(__name__)
    logger.info("...")

The first call to get_logger() (from anywhere, in any module) creates
backend/results/logs/logs_<YYYYMMDD_HHMMSS>.log and attaches a file handler
plus a console handler to a shared "mendxai" root logger. Every subsequent
call — from this module or any other — reuses the same handlers (guarded by
the `_configured` flag below), so multiple modules logging in the same
process all land in one file without duplicate lines or duplicate log files.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import config

_configured = False
_log_path: Optional[Path] = None


def _configure_once() -> None:
    global _configured, _log_path
    if _configured:
        return

    config.ensure_output_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = config.training.logs_dir / f"logs_{timestamp}.log"

    root_logger = logging.getLogger("mendxai")
    root_logger.setLevel(logging.INFO)
    root_logger.propagate = False

    file_handler = logging.FileHandler(_log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(stream_handler)

    _configured = True


def get_logger(name: str = "mendxai") -> logging.Logger:
    """Return a logger that writes to the shared, timestamped, process-wide
    log file (and to stdout). Safe to call from any module, any number of
    times — the underlying handlers are configured exactly once per process."""
    _configure_once()
    return logging.getLogger("mendxai" if name in ("mendxai", "__main__") else f"mendxai.{name}")


def get_log_path() -> Path:
    """Path to the current run's log file. Triggers logger configuration if
    it hasn't happened yet (so this is always valid to call)."""
    _configure_once()
    return _log_path
