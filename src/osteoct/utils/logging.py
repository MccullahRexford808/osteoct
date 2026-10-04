"""Logging configuration for the command-line entry points."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "osteoct"
DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_level(level))
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_logger(component: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{component}")


def _level(name: str) -> int:
    resolved = logging.getLevelName(name.upper())
    return resolved if isinstance(resolved, int) else logging.INFO
