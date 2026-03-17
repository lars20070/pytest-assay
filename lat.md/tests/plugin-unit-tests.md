---
lat:
  require-code-mention: true
---
# Plugin Unit Tests

Helper and hook coverage for `src/pytest_assay/plugin.py` in `tests/test_plugin.py`.

## _path

Tests for the `_path()` helper that computes `<test_dir>/assays/<module_stem>/<test_name>.json`.

- Standard path: verifies the path is absolute, ends with `.json`, and contains the `assays` directory.
- Parametrized test name: verifies that `[param1-param2]` suffixes are stripped so parametrized variants share the same file name root.
- Nested directory: verifies that deeply nested test module paths resolve correctly.

## _is_assay

Tests for the `_is_assay()` guard that allows only `Function` items with the `assay` marker to proceed.

- Returns `True` for a `Function` item with the `assay` marker present.
- Returns `False` for a `Function` item where `get_closest_marker` returns `None`.
- Returns `False` for a non-`Function` item (e.g., a plain `Item`) regardless of markers.

## pytest_addoption

Verifies that `pytest_addoption` calls `parser.addoption` exactly once with `--assay-mode`, default `"evaluate"`, and choices `("evaluate", "new_baseline")`.

## pytest_configure

Verifies that `pytest_configure` calls `config.addinivalue_line` with `"markers"` and a string containing `"assay"`.

## pytest_runtest_setup

Tests for the setup hook that loads the dataset and injects `AssayContext`.

- Skips non-assay items without touching `funcargs`.
- Loads an existing dataset file: verifies `BASELINE_DATASET_KEY` is stashed, `AssayContext` is injected with the correct dataset, path, and mode, and case content is intact.
- Calls the generator when no file exists: verifies the generator is called once, the dataset file is created and reloadable, and `AssayContext` reflects the generated cases.
- Raises `TypeError` when the generator returns a non-`Dataset` value.
- Falls back to an empty `Dataset(cases=[])` when neither a file nor a generator is provided.
- Baseline is a deep copy: mutating `assay_ctx.dataset` does not change `item.stash[BASELINE_DATASET_KEY]`.

## pytest_runtest_call

Tests for the hookwrapper that monkeypatches `Agent.run()`.

- Yields immediately (no patching) for non-`Function` items.
- Yields immediately for `Function` items without the `assay` marker.
- Initializes `item.stash[AGENT_RESPONSES_KEY]` to `[]` before yielding.
- Sets `_current_item_var` to the current item while inside the yield; resets it to `None` after.
- `Agent.run` is replaced by the instrumented wrapper during the yield, then restored to the original method in the `finally` block.
- Calling the instrumented `Agent.run` captures the result in `AGENT_RESPONSES_KEY` and passes the return value through to the caller without infinite recursion.
- When `_current_item_var` is manually cleared to `None`, the instrumented wrapper still calls the original and returns the result but does not append to any stash.

## pytest_runtest_teardown — new_baseline path

Tests for teardown when `assay_mode == "new_baseline"`.

- Skips non-assay items without side effects.
- Handles missing `AssayContext` in `funcargs` gracefully (no exception).
- Merges captured responses into `case.expected_output` and serializes the dataset to disk; verifies the JSON file is reloadable and contains the correct `expected_output` values.
- Skips serialization and logs an error when the number of captured responses does not match the number of dataset cases.
- Writes an empty string for `None` response output instead of failing.
- Skips serialization and logs an error when the call phase ran but captured zero responses against a non-empty dataset.

## pytest_runtest_teardown — evaluate path

Tests for teardown when `assay_mode == "evaluate"`.

- Does not write a dataset file; only writes a `.readout.json` file.
- Calls the evaluator with an `EvaluatorInput` and serializes the returned `Readout` to `<assay_path>.readout.json`; verifies file content.
- Uses `BradleyTerryEvaluator()` as the default evaluator when none is specified in marker kwargs.
- Logs an error and returns early for a non-callable evaluator value.
- Catches exceptions from the evaluator and logs them via `logger.exception` without propagating.

## Full Workflow

End-to-end test covering setup → call → teardown in `new_baseline` mode.

A generator creates cases, `pytest_runtest_setup` injects the context, simulated `Agent.run()` responses are stored in the stash, and teardown serializes the merged dataset with the correct `expected_output` per case.
