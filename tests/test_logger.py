#!/usr/bin/env python3

import logfire
from loguru import logger as loguru_base_logger

from pytest_assay.logger import logger


def test_logger_is_loguru_instance() -> None:
    """Test that logger is a Loguru logger."""
    assert type(logger) is type(loguru_base_logger)


def test_loguru_file_handler() -> None:
    """Test that Loguru has a file handler configured."""
    # The default handler (id=0) was removed, so a file handler should exist
    handlers = logger._core.handlers  # type: ignore[attr-defined]
    assert len(handlers) > 0

    # At least one handler should write to a file with the expected name
    sink_paths = [getattr(h._sink, "_path", "") for h in handlers.values()]  # type: ignore[attr-defined]
    assert any("pytest-assay.log" in str(p) for p in sink_paths)


def test_loguru_logging() -> None:
    """Test that Loguru can write log messages without error."""
    logger.info("Test message from test_loguru_logging")
    logger.debug("Debug message from test_loguru_logging")


def test_logfire_logging() -> None:
    """Test that Logfire can write log messages without error."""
    logfire.info("Test message from test_logfire_logging")
