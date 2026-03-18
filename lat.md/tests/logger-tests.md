---
lat:
  require-code-mention: true
---
# Logger Tests

Tests in [[tests/test_logger.py]] for the `pytest_assay` logger.

## Is Stdlib Logger

The `logger` export is an instance of `logging.Logger`, not a third-party logger type. See [[tests/test_logger.py#test_logger_is_stdlib_logger]].

## Logger Name

The logger's `name` attribute is `"pytest_assay"`, matching the package name. See [[tests/test_logger.py#test_logger_name]].

## Null Handler Registered

A `NullHandler` is attached to the `pytest_assay` root logger — library best practice that suppresses all output unless the host application configures a handler. See [[tests/test_logger.py#test_null_handler_registered]].

## Logging Does Not Raise

Calling `logger.info()` and `logger.debug()` with a message string does not raise any exception. See [[tests/test_logger.py#test_logging_does_not_raise]].
