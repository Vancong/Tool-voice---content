# -*- coding: utf-8 -*-
"""
src/utils/logger.py

Centralised logging configuration for the AI Movie Review pipeline.

Provides a single ``get_logger(name)`` function that returns a loguru logger
bound with a ``job_id`` context variable.  All modules call this function
instead of instantiating loggers directly.

Supports verbose debug mode and job-specific file logging under logs/YYYY-MM-DD/job_<job_id>.log.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger as _root_logger

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
_LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_FILE: str = os.getenv("LOG_FILE", "logs/movie_review.log")

_log_path = Path(_LOG_FILE)
_log_path.parent.mkdir(parents=True, exist_ok=True)

_root_logger.remove()

# Main default file handler – rotating, async-safe.
_main_file_handler_id = _root_logger.add(
    str(_log_path),
    level=_LOG_LEVEL,
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
        "{extra[job_id]} | {name} | {message}"
    ),
    rotation="10 MB",
    retention="10 days",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

# stderr handler – coloured, for interactive use (only if sys.stderr is not None).
if sys.stderr is not None:
    _stderr_handler_id = _root_logger.add(
        sys.stderr,
        level=_LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{extra[job_id]}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

_default_logger = _root_logger.bind(job_id="main")

_active_job_handler_id: Optional[int] = None


def set_debug_mode(enabled: bool) -> None:
    """Set log level dynamically for future log messages."""
    global _LOG_LEVEL
    _LOG_LEVEL = "DEBUG" if enabled else "INFO"
    os.environ["LOG_LEVEL"] = _LOG_LEVEL


def setup_job_file_logger(job_id: str, debug_mode: bool = False) -> Path:
    """Create a dedicated log file at logs/YYYY-MM-DD/job_<job_id>.log.

    Parameters
    ----------
    job_id: str
        Unique job identifier.
    debug_mode: bool
        If True, enables verbose DEBUG logging to the job file.

    Returns
    -------
    Path
        Path to the created job log file.
    """
    global _active_job_handler_id
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path("logs") / today_str
    log_dir.mkdir(parents=True, exist_ok=True)
    job_log_path = log_dir / f"job_{job_id}.log"

    level = "DEBUG" if debug_mode else "INFO"

    if _active_job_handler_id is not None:
        try:
            _root_logger.remove(_active_job_handler_id)
        except Exception:
            pass

    _active_job_handler_id = _root_logger.add(
        str(job_log_path),
        level=level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
            "{extra[job_id]} | {name} | {message}"
        ),
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    return job_log_path.resolve()


def get_logger(name: str = "main"):
    """Return a loguru logger bound with *name* as the ``job_id`` context."""
    return _root_logger.bind(job_id=name)


__all__ = ["get_logger", "setup_job_file_logger", "set_debug_mode"]
