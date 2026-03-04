import logging

from pytest_assay.logger import logger


def test_logger_is_stdlib_logger() -> None:
    """Test that logger is a stdlib logging.Logger."""
    assert isinstance(logger, logging.Logger)


def test_logger_name() -> None:
    """Test that the logger has the correct name."""
    assert logger.name == "pytest_assay"


def test_null_handler_registered() -> None:
    """Test that NullHandler is registered on the pytest_assay logger."""
    root = logging.getLogger("pytest_assay")
    assert any(isinstance(h, logging.NullHandler) for h in root.handlers)


def test_logging_does_not_raise() -> None:
    """Test that logging calls do not raise."""
    logger.info("Test message from test_logging_does_not_raise")
    logger.debug("Debug message from test_logging_does_not_raise")
