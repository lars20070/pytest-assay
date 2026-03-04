# Pytest plugins should never configure logging at import time

**The universal standard among well-known pytest plugins is clear: none use loguru, logfire, or even Python's stdlib `logging` module for their own internal diagnostics.** All eight major plugins examined — pytest-xdist, pytest-cov, pytest-django, pytest-asyncio, pytest-mock, pytest-html, pytest-timeout, and pytest-randomly — defer configuration entirely to pytest hooks and rely on pytest's own reporting infrastructure for output. A pytest plugin that removes loguru's default handler, adds a file handler, and calls `logfire.configure()` at module import time violates every established convention in the ecosystem and will interfere with users' logging configuration, the `caplog` fixture, and pytest's built-in `--log-level`/`--log-file` options.

## No major pytest plugin uses Python's logging module internally

The most striking finding from examining source code across eight major plugins is that **none of them call `logging.getLogger()`** for their own operational output. Instead, they use lighter-weight mechanisms native to pytest's architecture:

- **pytest-xdist** uses a custom `Producer` class that wraps `print()` to `sys.stderr`, gated behind `config.option.debug` so output only appears when the user passes `--debug`
- **pytest-cov** uses `warnings.warn()` with a custom `CovReportWarning` class and the terminal reporter's `write()` method for user-facing messages
- **pytest-asyncio, pytest-mock, and pytest-html** all use `warnings.warn()` for deprecation notices and error conditions
- **pytest-randomly** reports its random seed through the `pytest_report_header` hook
- **pytest-timeout** dumps stack traces directly to `sys.stdout`/`sys.stderr` when timeouts fire

This pattern exists for a reason. By avoiding Python's `logging` module entirely, these plugins **cannot interfere with pytest's log capture system**, which installs handlers on the root logger via the `catching_logs` context manager during every test phase. Any plugin that adds its own handlers to the logging hierarchy risks conflicting with `caplog`, `--log-cli-level`, and `--log-file`.

## Python's official guidance prohibits handler configuration in libraries

The Python documentation contains explicit, unambiguous guidance in its "Configuring Logging for a Library" section. The canonical pattern for any library or plugin is exactly two lines:

```python
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
```

The docs state: **"It is strongly advised that you do not add any handlers other than NullHandler to your library's loggers. This is because the configuration of handlers is the prerogative of the application developer who uses your library."** They further warn: "if you add handlers 'under the hood', you might well interfere with their ability to carry out unit tests and deliver logs which suit their requirements."

This principle applies doubly to pytest plugins. The "application developer" in this context is both pytest itself (which has a sophisticated logging infrastructure) and the end user (who configures logging through pytest's CLI options and `pytest.ini`). A plugin that adds file handlers or configures external services takes control away from both.

The docs also explicitly prohibit logging to the root logger, setting log levels in library code, and defining custom log levels — all of which "will make it difficult or impossible for the application developer to configure the logging verbosity or handlers of your library as they wish."

## Five specific anti-patterns in the current pytest-assay approach

The described `logger.py` module in pytest-assay exhibits several well-documented anti-patterns that compound each other:

**Removing and adding loguru handlers at module level** is the most damaging pattern. Loguru uses a single global logger, so `logger.remove()` destroys handlers that other libraries or the user have configured, and `logger.add("pytest-assay.log")` creates a sink that captures log messages from *all* modules using loguru — not just pytest-assay. Even loguru's own documentation explicitly warns: "If your work is intended to be used as a library, you usually should not add any handler" and "never call `add()`" from library code. The only acceptable loguru pattern for libraries is `logger.disable("mylib")` in `__init__.py`.

**Calling `logfire.configure()` at module level** initializes Pydantic Logfire's entire OpenTelemetry infrastructure — span processors, exporters, and tracer providers — as a global side effect of importing the module. Even though Logfire detects `PYTEST_VERSION` and sets `send_to_logfire=False` during tests, it still mutates global OpenTelemetry state before pytest's own logging infrastructure has initialized. This means any module that does `from pytest_assay import something` triggers OpenTelemetry setup, adding latency and creating state that persists across the test session.

**Creating "pytest-assay.log" unconditionally** means a log file appears in the working directory every time any code imports from pytest-assay, even if the user never invoked any pytest-assay functionality. This pollutes the filesystem, may fail with permission errors in read-only environments (CI containers, for instance), and duplicates functionality that pytest already provides through `--log-file`.

**These side effects execute at import time**, which means they happen during pytest's collection phase, before the user has any opportunity to configure behavior through command-line options or `pytest.ini` settings. PEP 690 (Lazy Imports) identifies import side effects as a category of bugs, and real-world debugging case studies document how import-time side effects in pytest cause tests to fail when run together but pass individually.

**Loguru fundamentally bypasses pytest's log capture.** Because loguru does not use stdlib `logging` internally, the `caplog` fixture cannot capture loguru messages without a custom fixture override. Similarly, `--log-cli-level`, `--log-level`, and `--log-file` have zero effect on loguru output since they control stdlib logging handlers. Users who expect standard pytest logging behavior will find that pytest-assay's logs are invisible to `caplog` and uncontrollable through standard CLI options.

## How pytest's logging infrastructure works and how plugins should integrate

Pytest's built-in `LoggingPlugin` (registered as `"logging-plugin"`) provides a complete logging system that plugins should leverage rather than replace. It registers during `pytest_configure` with `@hookimpl(trylast=True)` and implements hook wrappers around every test lifecycle phase — `pytest_runtest_setup`, `pytest_runtest_call`, `pytest_runtest_teardown`, and session-level hooks.

The core mechanism is a `catching_logs` context manager that attaches a `LogCaptureHandler` to the **root stdlib logger** for each test phase, captures all `logging.LogRecord` instances, and adds them to test reports via `item.add_report_section()`. This means any plugin using `logging.getLogger(__name__)` gets automatic integration with:

- **`caplog` fixture**: Users can assert on log messages with `caplog.records`, `caplog.text`, and `caplog.record_tuples`
- **`--log-level`**: Controls the minimum severity captured globally
- **`--log-cli-level`**: Enables live log output to the terminal during test execution
- **`--log-file` and `--log-file-level`**: Routes all captured logs to a file the user specifies, with configurable severity filtering
- **`--log-format` and `--log-date-format`**: Controls output formatting across all log destinations

Plugins that need to interact with this system should access it through `config.pluginmanager.get_plugin("logging-plugin")` in a `pytest_configure` hook. The logging plugin exposes `log_cli_handler`, `log_file_handler`, and a `set_log_path()` method (marked experimental). This is the only documented plugin-to-logging integration point.

A critical timing consideration: because `LoggingPlugin` registers with `trylast=True`, plugins that need the logging infrastructure available must use `@pytest.hookimpl(trylast=True)` on their own `pytest_configure` or run in a subsequent hook.

## The correct pattern for a pytest plugin that needs logging

If pytest-assay needs internal diagnostic logging, the recommended refactoring follows established conventions from both Python's official documentation and the patterns used by every major pytest plugin:

```python
# pytest_assay/__init__.py
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
```

```python
# pytest_assay/plugin.py
import logging

logger = logging.getLogger(__name__)

def pytest_configure(config):
    """All setup deferred to this hook — never at import time."""
    logger.debug("pytest-assay plugin configured")
    # If logfire integration is desired, it should be opt-in:
    if config.getoption("--assay-logfire", default=False):
        import logfire
        logfire.configure()

def pytest_unconfigure(config):
    """Clean up any resources."""
    logger.debug("pytest-assay plugin unconfigured")
```

This approach gives users full control. Their `--log-level=DEBUG` will surface pytest-assay's debug messages. Their `--log-file=test.log` will capture them to a file. Their `caplog` assertions will work without custom fixture overrides. And if they never use pytest-assay's features, zero side effects occur.

## Conclusion

The ecosystem consensus is unambiguous. **Every major pytest plugin avoids configuring logging at import time**, and most avoid Python's `logging` module entirely in favor of pytest's native reporting hooks. Python's official documentation explicitly reserves handler configuration for application code, not libraries. Loguru's own maintainers warn against calling `logger.add()` in library code. The current pytest-assay approach of configuring loguru sinks and calling `logfire.configure()` at module level creates unconditional file I/O, breaks `caplog` compatibility, ignores pytest's CLI logging options, and mutates global state before the test framework initializes. The fix is straightforward: switch to `logging.getLogger(__name__)` with `NullHandler`, defer all configuration to `pytest_configure`, and make external service integration like Logfire explicitly opt-in through command-line options.