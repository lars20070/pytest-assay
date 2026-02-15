#!/usr/bin/env python3

import logfire
from loguru import logger

__all__ = ["logger"]

from .config import config

# Configure Logfire
logfire.configure(
    token=config.logfire_token,
    send_to_logfire=True,
    scrubbing=False,
)

# Configure Loguru
logger.remove(0)  # Remove default console logger
logger.add(
    "pytest-assay.log",
    rotation="500 MB",
    level="DEBUG",
)
