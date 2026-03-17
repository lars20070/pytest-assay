---
lat:
  require-code-mention: true
---
# Pytest Plugin

The pytest-assay plugin hooks into the pytest lifecycle to intercept `Agent.run()` calls, capture model outputs, and trigger evaluation or baseline serialization after each test.

## Marker

The `@pytest.mark.assay` marker designates a test function as an assay. It accepts two optional keyword arguments:

- `generator` — a callable returning a `Dataset` for building test cases when no baseline file exists yet.
- `evaluator` — an `Evaluator` instance controlling the comparison strategy. Defaults to `BradleyTerryEvaluator()`.

```python
@pytest.mark.assay(
    generator=my_dataset_generator,
    evaluator=BradleyTerryEvaluator(criterion="Which response is more accurate?"),
)
async def test_my_agent(context: AssayContext) -> None:
    result = await agent.run("some prompt")
    ...
```

Only `Function` items (not classes or modules) are treated as valid assay tests. See [[pytest-plugin#_is_assay]].

## Assay Modes

Controlled via the `--assay-mode` CLI option. Two modes are supported:

- `evaluate` (default) — loads the baseline dataset, runs the evaluator against captured responses, and writes a `.readout.json` report.
- `new_baseline` — merges captured responses into the dataset cases as `expected_output` and serializes to disk, replacing the previous baseline.

The mode is stored on the `AssayContext` injected into each test. See [[pytest-plugin#Lifecycle]].

## Baseline File Layout

Baseline datasets are stored as JSON files under the test directory:

```
<test_dir>/assays/<module_stem>/<test_name>.json
```

For example, `tests/test_agents.py::test_summariser` → `tests/assays/test_agents/test_summariser.json`.

The path is computed by [[pytest-plugin#_path]] and stored on `AssayContext.path`.

## Lifecycle

The plugin participates in three pytest hook phases for every assay test:

### Setup — pytest_runtest_setup

Runs before the test body. Priority: `tryfirst`.

1. Computes the baseline file path via `_path(item)`.
2. If the file exists, loads it as a `Dataset`.
3. If no file exists but a `generator` was provided, calls it to create and immediately persist the dataset.
4. Otherwise starts with an empty `Dataset(cases=[])`.
5. Takes a deep copy as the immutable baseline snapshot → `item.stash[BASELINE_DATASET_KEY]`.
6. Injects an `AssayContext` (mutable dataset + path + mode) as `item.funcargs["context"]`.

### Call — pytest_runtest_call

Runs around the test body. Uses `hookwrapper=True`.

Monkeypatches `Agent.run` with `_instrumented_agent_run` for the duration of the test:

1. Saves `Agent.run` reference before patching.
2. Pushes the current `Item` into `_current_item_var` (a module-level `ContextVar`).
3. Replaces `Agent.run` with a wrapper that awaits the original, then appends the result to `item.stash[AGENT_RESPONSES_KEY]`.
4. After `yield`, restores the original `Agent.run` and resets the `ContextVar` token.

The `ContextVar` approach allows the wrapper closure to locate the correct test item without modifying `Agent.run`'s signature. See [[pytest-plugin#ContextVar Tunnel]].

### Teardown — pytest_runtest_teardown

Runs before fixture finalization (`yield` deferred). Checks that `AGENT_RESPONSES_KEY` is present (absent means the test was skipped).

- `new_baseline` mode → calls `_serialize_baseline(item, assay)`.
- `evaluate` mode → calls `_run_evaluation(item, assay)`.

Fixture teardown (e.g. VCR cassette close) happens after `yield`.

## ContextVar Tunnel

`_current_item_var: ContextVar[Item | None]` is defined at module level. Because `Agent.run` is an async method called inside async test functions, a plain closure variable would be unsafe under concurrent execution. The `ContextVar` provides an execution-context-scoped binding: each asyncio task sees its own value, preventing cross-test contamination.

The call-hook sets the var with `set()`, which returns a `Token`. The `finally` block calls `reset(token)` to cleanly unwind, even if the test raises.

## _path

[[src/pytest_assay/plugin.py#_path]]

Computes `<test_dir>/assays/<module_stem>/<test_name>.json`. Strips parametrize suffixes (`[param]`) from the test name before constructing the path so parametrized variants share a common baseline file name root.

## _is_assay

[[src/pytest_assay/plugin.py#_is_assay]]

Returns `True` only when the item is a `Function` and carries the `assay` marker. Guards all hook logic to avoid interfering with non-assay tests.

## _serialize_baseline

[[src/pytest_assay/plugin.py#_serialize_baseline]]

Merges `item.stash[AGENT_RESPONSES_KEY]` into `assay.dataset.cases` as `expected_output` strings. Requires an exact 1-to-1 match between response count and case count; logs an error and skips serialization if they differ. Writes via `Dataset.to_file()` with `schema_path=None`.

## _run_evaluation

[[src/pytest_assay/plugin.py#_run_evaluation]]

Retrieves the `evaluator` callable from the marker kwargs (default: `BradleyTerryEvaluator()`). Builds an `EvaluatorInput` from the baseline snapshot and captured responses, then calls `asyncio.run(evaluator(eval_input))`. Serializes the resulting `Readout` to `<assay_path>.readout.json`. See `lat.md/evaluators.md` for evaluator contracts.

## Stash Keys

| Key | Type | Set by | Read by |
|-----|------|--------|---------|
| `BASELINE_DATASET_KEY` | `Dataset` | `pytest_runtest_setup` | `_run_evaluation` |
| `AGENT_RESPONSES_KEY` | `list[AgentRunResult]` | `pytest_runtest_call` | `_serialize_baseline`, `_run_evaluation`, `pytest_runtest_teardown` |
