# MIT License
# src/utils/logger.py
"""
Logger helper used across modules.
"""

from __future__ import annotations
import logging
import sys
from typing import Optional

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if logger.handlers:
        # avoid adding multiple handlers in iterative runs
        logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger

if __name__ == "__main__":
    log = get_logger("logger_test")
    log.info("Logger is working")