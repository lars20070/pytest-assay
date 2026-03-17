---
lat:
  require-code-mention: true
---
# Logger Tests

Tests in `tests/test_logger.py` for the `pytest_assay` logger.

- `logger` is an instance of `logging.Logger`.
- Logger name is `"pytest_assay"`.
- A `NullHandler` is registered on the `pytest_assay` root logger (library best practice — no output unless the host configures it).
- `logger.info()` and `logger.debug()` calls do not raise.
