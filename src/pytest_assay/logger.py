import logging

logger = logging.getLogger("pytest_assay")
logger.addHandler(logging.NullHandler())

__all__ = ["logger"]
