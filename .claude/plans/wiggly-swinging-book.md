# Logging Redesign: Remove Loguru/Logfire, Use stdlib logging

## Context
Current `logger.py` violates pytest plugin best practices: removes loguru handlers at import time, calls `logfire.configure()` globally, and creates a log file unconditionally. Per `.claude/plans/logging_bestpractice.md`, the fix is stdlib `logging.getLogger()` with `NullHandler`. Remove `loguru` and `logfire` dependencies entirely.

## Changes

### 1. `src/pytest_assay/logger.py` — Rewrite
```python
import logging

logger = logging.getLogger("pytest_assay")
__all__ = ["logger"]
```
No handlers, no file creation, no global state mutation. All 50+ call sites keep identical API (`.debug/.info/.warning/.error`).

### 2. `src/pytest_assay/__init__.py` — Add NullHandler
```python
import logging
logging.getLogger("pytest_assay").addHandler(logging.NullHandler())
```
Prevents "No handlers could be found" warnings when users don't configure logging.

### 3. `pyproject.toml` — Remove dependencies
- Remove `loguru>=0.7.3`
- Remove `logfire[httpx]>=3.10.0`
- Remove `pydantic-evals[logfire]>=0.0.51` → `pydantic-evals>=0.0.51`

### 4. `src/pytest_assay/config.py` — Remove logfire_token field
Drop `logfire_token` from `AssayConfig` (no longer needed).

### 5. `tests/test_logger.py` — Replace loguru test
Replace test that checks loguru behavior with a stdlib logging test (e.g., assert `logger.name == "pytest_assay"`, assert `NullHandler` is registered).

### 6. All other call sites — No changes needed
`plugin.py`, `evaluators/bradleyterry.py`, `evaluators/pairwise.py`, `tests/` all keep identical `logger.debug/info/warning/error()` calls. stdlib logging has the same API.

## Behavior after change
- Default: zero output (NullHandler swallows everything)
- `pytest --log-cli-level=DEBUG`: streams all debug to terminal live
- `pytest --log-level=INFO`: captures info+ in test reports
- `caplog` fixture: works natively without workarounds
- No files created, no global state mutated at import time

## Files to modify
- [src/pytest_assay/logger.py](src/pytest_assay/logger.py)
- [src/pytest_assay/__init__.py](src/pytest_assay/__init__.py)
- [src/pytest_assay/config.py](src/pytest_assay/config.py) — remove `logfire_token`
- [pyproject.toml](pyproject.toml) — remove loguru/logfire deps
- [tests/test_logger.py](tests/test_logger.py) — update test

## Verification
```bash
uv sync                          # lock file updates, loguru/logfire removed
uv run ruff check --fix .
uv run pyright .
uv run pytest -n auto            # all tests pass
uv run pytest --log-cli-level=DEBUG tests/test_logger.py -v  # debug output visible
```

## Unresolved questions
- Does `pydantic-evals` use `logfire` internally in a way that breaks if we remove `[logfire]` extra? Check after `uv sync`.
