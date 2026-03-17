---
lat:
  require-code-mention: true
---
# Tests

Test suite for pytest-assay. Unit tests use `pytest-mock` and `AsyncMock`; integration tests require a local Ollama instance and are gated by `@pytest.mark.ollama`.

## Plugin Unit Tests

[[src/pytest_assay/plugin.py]] helper and hook coverage in `tests/test_plugin.py`.

### _path

Tests for the `_path()` helper that computes `<test_dir>/assays/<module_stem>/<test_name>.json`.

- Standard path: verifies the path is absolute, ends with `.json`, and contains the `assays` directory.
- Parametrized test name: verifies that `[param1-param2]` suffixes are stripped so parametrized variants share the same file name root.
- Nested directory: verifies that deeply nested test module paths resolve correctly.

### _is_assay

Tests for the `_is_assay()` guard that allows only `Function` items with the `assay` marker to proceed.

- Returns `True` for a `Function` item with the `assay` marker present.
- Returns `False` for a `Function` item where `get_closest_marker` returns `None`.
- Returns `False` for a non-`Function` item (e.g., a plain `Item`) regardless of markers.

### pytest_addoption

Verifies that `pytest_addoption` calls `parser.addoption` exactly once with `--assay-mode`, default `"evaluate"`, and choices `("evaluate", "new_baseline")`.

### pytest_configure

Verifies that `pytest_configure` calls `config.addinivalue_line` with `"markers"` and a string containing `"assay"`.

### pytest_runtest_setup

Tests for the setup hook that loads the dataset and injects `AssayContext`.

- Skips non-assay items without touching `funcargs`.
- Loads an existing dataset file: verifies `BASELINE_DATASET_KEY` is stashed, `AssayContext` is injected with the correct dataset, path, and mode, and case content is intact.
- Calls the generator when no file exists: verifies the generator is called once, the dataset file is created and reloadable, and `AssayContext` reflects the generated cases.
- Raises `TypeError` when the generator returns a non-`Dataset` value.
- Falls back to an empty `Dataset(cases=[])` when neither a file nor a generator is provided.
- Baseline is a deep copy: mutating `assay_ctx.dataset` does not change `item.stash[BASELINE_DATASET_KEY]`.

### pytest_runtest_call

Tests for the hookwrapper that monkeypatches `Agent.run()`.

- Yields immediately (no patching) for non-`Function` items.
- Yields immediately for `Function` items without the `assay` marker.
- Initializes `item.stash[AGENT_RESPONSES_KEY]` to `[]` before yielding.
- Sets `_current_item_var` to the current item while inside the yield; resets it to `None` after.
- `Agent.run` is replaced by the instrumented wrapper during the yield, then restored to the original method in the `finally` block.
- Calling the instrumented `Agent.run` captures the result in `AGENT_RESPONSES_KEY` and passes the return value through to the caller without infinite recursion.
- When `_current_item_var` is manually cleared to `None`, the instrumented wrapper still calls the original and returns the result but does not append to any stash.

### pytest_runtest_teardown — new_baseline path

Tests for teardown when `assay_mode == "new_baseline"`.

- Skips non-assay items without side effects.
- Handles missing `AssayContext` in `funcargs` gracefully (no exception).
- Merges captured responses into `case.expected_output` and serializes the dataset to disk; verifies the JSON file is reloadable and contains the correct `expected_output` values.
- Skips serialization and logs an error when the number of captured responses does not match the number of dataset cases.
- Writes an empty string for `None` response output instead of failing.
- Skips serialization and logs an error when the call phase ran but captured zero responses against a non-empty dataset.

### pytest_runtest_teardown — evaluate path

Tests for teardown when `assay_mode == "evaluate"`.

- Does not write a dataset file; only writes a `.readout.json` file.
- Calls the evaluator with an `EvaluatorInput` and serializes the returned `Readout` to `<assay_path>.readout.json`; verifies file content.
- Uses `BradleyTerryEvaluator()` as the default evaluator when none is specified in marker kwargs.
- Logs an error and returns early for a non-callable evaluator value.
- Catches exceptions from the evaluator and logs them via `logger.exception` without propagating.

### Full Workflow

End-to-end test covering setup → call → teardown in `new_baseline` mode: a generator creates cases, `pytest_runtest_setup` injects the context, simulated `Agent.run()` responses are stored in the stash, and teardown serializes the merged dataset with the correct `expected_output` per case.

## Model Unit Tests

Tests for data models in `tests/test_models.py`.

### EvaluatorInput

- Creation with `baseline_dataset=None` and a single response.
- Creation with a populated `Dataset` and verification of case count and names.
- Empty `agent_responses` list accepted.
- Multiple responses stored in order.
- `ValidationError` raised when all required fields are omitted.
- `ValidationError` raised when `agent_responses` is omitted.
- `ValidationError` raised when `baseline_dataset` is omitted.
- Exactly two model fields (`baseline_dataset`, `agent_responses`) are defined.
- A response with `output=None` is stored without error.

### Readout

- Default values: `passed=True`, `details=None`.
- Custom values accepted for both fields.
- `model_dump()` returns the correct dict for custom and default values.
- `to_file()` writes valid JSON; verifies `passed` and `details` fields in the file.
- `to_file()` with `details=None` writes `null`.
- `to_file()` with `details={}` writes an empty object.

### Evaluator Protocol

- `Evaluator` is callable (protocol is importable and callable as a type).
- Protocol has the expected structure (`__protocol_attrs__` or callable).

### AssayContext

- Created with all required fields; verifies attribute values.
- Default `assay_mode` is `"evaluate"`.
- `assay_mode="new_baseline"` accepted.
- Dataset with cases: verifies case count, field access, mutability (cases can be cleared and extended).
- `ValidationError` raised when required fields are omitted.

## BradleyTerry Evaluator Tests

Unit and integration tests in `tests/evaluators/test_bradleyterry.py`.

### EvalPlayer

- Created with required fields; `score` defaults to `None`.
- Created with explicit `score`.
- `score` is mutable after creation.
- Negative `idx` is permitted (it is an identifier, not a position).
- Empty `item` string is permitted.

### EvalGame

- `criterion` is stored on the model.
- Returns `(player_A_idx, player_B_idx)` when the agent picks `"A"`.
- Returns `(player_B_idx, player_A_idx)` when the agent picks `"B"`.
- The prompt sent to the agent contains both player items and the criterion string.
- Result tuple uses actual `idx` values, not positional indices.

### EvalTournament

- Constructed with the correct number of players and the game criterion.
- `get_player_by_idx` returns the player with the matching `idx` and correct `item`.
- `get_player_by_idx` raises `ValueError` for an unknown `idx`.
- When no strategy is provided, `adaptive_uncertainty_strategy` is invoked.
- A custom strategy function receives `(players, game, agent, model_settings)` positional args.
- Extra kwargs (e.g., `max_standard_deviation`, `alpha`) are forwarded to the strategy.
- After `run()`, `tournament.players` is updated to the scored players returned by the strategy.

### BradleyTerryEvaluator

- Default init: `OpenAIChatModel` on Ollama, `temperature=0.0`, `timeout=300`, default `criterion` and `max_standard_deviation=2.0`.
- Custom `criterion` and `max_standard_deviation` stored.
- Custom `OpenAIChatModel` instance is used and reflected on the internal agent.
- Model string (e.g., `"openai:gpt-4o-mini"`) accepted.
- Empty player list returns `Readout(passed=True, details={"message": "No players to evaluate"})`.
- With one baseline case and one novel response, creates a tournament with the correct criterion and `max_standard_deviation`, returns a `Readout`.
- Conforms to `Evaluator` protocol: callable, `__call__` is a coroutine function.

### BradleyTerry Strategy Integration (ollama)

Requires Ollama. All integration tests are marked `@pytest.mark.ollama`.

- `EvalGame` integration: running a game between vanilla and a creative ice cream flavour returns the expected winner index.
- `EvalTournament` integration: running the default strategy scores all players with non-`None` float scores; running `random_sampling_strategy` with `fraction_of_games=0.3` also scores all players.
- `random_sampling_strategy` parametrized over `fraction_of_games=None`, `0.3`, and an out-of-range value; all produce float scores for every player.
- `round_robin_strategy` with `number_of_rounds=1` produces float scores for every player.
- `adaptive_uncertainty_strategy` with `max_standard_deviation=1.0` and `alpha=0.01` produces float scores for every player.

## Pairwise Evaluator Tests

Unit tests in `tests/evaluators/test_pairwise.py`.

### PairwiseEvaluator

- Default init: `OpenAIChatModel` on Ollama, `temperature=0.0`, `timeout=300`, default criterion.
- Custom criterion stored.
- Custom `OpenAIChatModel` used and reflected on the internal agent.
- Model string accepted.
- Empty baseline and novel lists: returns `Readout(passed=False)` with empty win lists.
- Novel wins all comparisons (`"B"` every time): `passed=True`, `wins_novel=[True, True]`.
- Baseline wins: `passed=False`, `wins_novel=[False]`.
- Tie (one win each): `passed=False` because novel must strictly exceed baseline.
- Raises `AssertionError` with `"Mismatch in response counts"` when baseline and novel counts differ.
- Novel responses with `None` output are skipped; if counts then match baseline (zero), no error.
- Prompt sent to agent contains the criterion, baseline text, and novel text.
- Conforms to `Evaluator` protocol: callable, `__call__` is a coroutine function.

## Config Tests

Tests in `tests/test_config.py` for the `Config` pydantic-settings model.

- Default values: `ollama_base_url="http://localhost:11434"`, `ollama_model="qwen2.5:14b"` when no env vars are set.
- Env override: `OLLAMA_BASE_URL` and `OLLAMA_MODEL` are read from environment variables.
- Case-insensitive: lowercase env var names are accepted.
- Extra env vars are silently ignored.
- Both fields have non-`None` descriptions.
- Module-level `config` instance is a `Config` with non-empty field values.

## Logger Tests

Tests in `tests/test_logger.py` for the `pytest_assay` logger.

- `logger` is an instance of `logging.Logger`.
- Logger name is `"pytest_assay"`.
- A `NullHandler` is registered on the `pytest_assay` root logger (library best practice — no output unless the host configures it).
- `logger.info()` and `logger.debug()` calls do not raise.

## Plugin Integration Tests

End-to-end tests against a live Ollama instance in `tests/test_plugin_integration.py`. All marked `@pytest.mark.ollama` and `@pytest.mark.asyncio`.

A shared `generate_evaluation_cases()` generator produces ten research-topic cases. An agent generates search queries using a creative prompt. Three evaluators are exercised:

- `test_integration_pairwiseevaluator`: uses `PairwiseEvaluator` with a creativity criterion; verifies the full plugin lifecycle produces a `.readout.json` on disk.
- `test_integration_bradleyterryevaluator`: uses `BradleyTerryEvaluator` with the same creativity criterion and `max_standard_deviation=2.1`.
- `test_integration_lengthevaluator`: uses a user-defined `LengthEvaluator` (not part of the package) that passes when a majority of novel responses are longer than their baseline counterparts; demonstrates the custom evaluator extension point.
