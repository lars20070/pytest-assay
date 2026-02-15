#!/usr/bin/env python3
import asyncio
import contextvars
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_evals import Dataset
from pytest import Config, Function, Item, Parser

from .evaluators.bradleyterry import BradleyTerryEvaluator  # noqa: F401
from .evaluators.pairwise import PairwiseEvaluator  # noqa: F401
from .logger import logger
from .models import AssayContext, Evaluator, Readout  # noqa: F401

# Modes for the assay plugin. "evaluate" is the default mode.
ASSAY_MODES = ("evaluate", "new_baseline")

# Key to stash the baseline dataset from previous runs
BASELINE_DATASET_KEY = pytest.StashKey[Dataset]()

# Key to stash Agent responses during assay tests
AGENT_RESPONSES_KEY = pytest.StashKey[list[AgentRunResult[Any]]]()

# Items stashed by the _wrapped_run wrapper. Required for async safety.
# _current_item_var defined at module level. But items are stored locally to the current execution context.
_current_item_var: contextvars.ContextVar[Item | None] = contextvars.ContextVar("_current_item", default=None)


# =============================================================================
# Helper functions for assay plugin
# =============================================================================


def _path(item: Item) -> Path:
    """
    Compute assay file path: <test_dir>/assays/<module>/<test>.json.

    Args:
        item: The pytest test item.
    """
    path = item.path
    module_name = path.stem
    test_name = item.name.split("[")[0]
    return path.parent / "assays" / module_name / f"{test_name}.json"


def _is_assay(item: Item) -> bool:
    """
    Check if item is a valid assay unit test:
    - item is a Function (not a class or module)
    - item has the @pytest.mark.assay marker

    Args:
        item: The pytest collection item to check.
    """
    if not isinstance(item, Function):
        return False
    return item.get_closest_marker("assay") is not None


def _serialize_baseline(item: Item, assay: AssayContext) -> None:
    """
    Serialize the dataset to disk in 'new_baseline' mode.

    Merges captured Agent.run() responses into dataset cases and writes to disk.

    Args:
        item: The pytest test item.
        assay: The assay context containing dataset and path.
    """
    # Merge captured responses into dataset cases
    responses = item.stash.get(AGENT_RESPONSES_KEY, [])
    cases = assay.dataset.cases

    if len(responses) != len(cases):
        logger.error(f"Cannot merge responses: {len(responses)} responses vs {len(cases)} cases. Skipping serialization.")
        return

    for case, response in zip(cases, responses, strict=True):
        case.expected_output = response.output if response.output is not None else ""

    logger.info(f"Serializing assay dataset to {assay.path}")
    assay.path.parent.mkdir(parents=True, exist_ok=True)
    assay.dataset.to_file(assay.path, schema_path=None)


def _run_evaluation(item: Item, assay: AssayContext) -> None:
    """
    Run the configured evaluator on captured responses in (default) 'evaluate' mode.

    Retrieves the evaluator from the marker, runs it asynchronously, and serializes the final readout report.

    Args:
        item: The pytest test item with captured responses.
        assay: The assay context containing dataset and path.
    """
    # Retrieve evaluator from marker kwargs with type validation
    marker = item.get_closest_marker("assay")
    assert marker is not None
    evaluator: Evaluator = marker.kwargs.get("evaluator", BradleyTerryEvaluator())
    if not callable(evaluator):
        logger.error(f"Invalid evaluator type: {type(evaluator)}. Expected callable.")
        return

    # Run the evaluator asynchronously
    try:
        readout = asyncio.run(evaluator(item))
        logger.info(f"Evaluation result: passed={readout.passed}")

        # Serialize the readout
        readout_path = assay.path.with_suffix(".readout.json")
        readout_path.parent.mkdir(parents=True, exist_ok=True)
        readout.to_file(readout_path)

    except Exception:
        logger.exception("Error during evaluation.")


# =============================================================================
# Pytest plugin hooks for assay functionality
# =============================================================================


def pytest_addoption(parser: Parser) -> None:
    """
    Register the --assay-mode option (evaluate or new_baseline).

    Args:
        parser: The pytest argument parser.
    """
    parser.addoption(
        "--assay-mode",
        action="store",
        default="evaluate",
        choices=ASSAY_MODES,
        help='Assay mode. Defaults to "evaluate".',
    )


def pytest_configure(config: Config) -> None:
    """
    Register the @pytest.mark.assay marker.

    Args:
        config: The pytest configuration object.
    """
    logger.info("Registering the @pytest.mark.assay marker.")
    config.addinivalue_line(
        "markers",
        "assay(generator=None, evaluator=BradleyTerryEvaluator()): "
        "Mark the test for AI agent evaluation (assay). "
        "Args: "
        "generator - optional callable returning a Dataset for test cases; "
        "evaluator - optional Evaluator instance for custom evaluation strategy "
        "(defaults to BradleyTerryEvaluator with default settings). "
        "Configure evaluators by instantiating with parameters: "
        "evaluator=BradleyTerryEvaluator(criterion='Which response is better?')",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: Item) -> None:
    """
    Load dataset and inject AssayContext into the test function.
    Loads from file if exists, else calls generator, else uses empty dataset.

    Args:
        item: The pytest test item being set up.
    """
    if not _is_assay(item):
        return

    # Get generator from marker kwargs
    marker = item.get_closest_marker("assay")
    assert marker is not None
    generator = marker.kwargs.get("generator")

    logger.info("Populating assay context with dataset and path")
    path = _path(item)
    if path.exists():
        logger.info(f"Loading assay dataset from {path}")
        dataset = Dataset[dict[str, str], str, Any].from_file(path)
    elif generator is not None:
        logger.info("Generating new assay dataset using custom generator")
        dataset = generator()

        if not isinstance(dataset, Dataset):
            raise TypeError(f"The generator {generator} must return a Dataset instance.")

        logger.info(f"Serialising generated assay dataset to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_file(path, schema_path=None)
    else:
        logger.info("No existing assay dataset file or generator found; using empty dataset")
        dataset = Dataset[dict[str, str], str, Any](cases=[])

    # Store immutable baseline snapshot for later evaluation
    item.stash[BASELINE_DATASET_KEY] = dataset.model_copy(deep=True)

    # Inject assay context into the test function arguments
    item.funcargs["context"] = AssayContext(  # type: ignore[attr-defined]
        dataset=dataset,
        path=path,
        assay_mode=item.config.getoption("--assay-mode"),
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: Item) -> Generator[None, None, None]:
    """
    Intercepts Agent.run() calls during test execution to record model outputs.

    Mechanism:
    1. Setup: Temporarily replaces Agent.run with an instrumented wrapper.
    2. Capture: When called, the wrapper retrieves the current test item via a ContextVar
       and saves the result to item.stash[AGENT_RESPONSES_KEY].
    3. Teardown: Restores the original Agent.run method.

    Why ContextVar?
    The Agent.run method signature cannot be modified to accept the test item.
    ContextVar acts as a thread-safe tunnel, allowing the wrapper to access the
    current test context implicitly without threading arguments through the call stack.
    """

    # Only inject for Function items i.e. actual test functions
    # For example, if @pytest.mark.assay decorates a class, we skip it here.
    if not isinstance(item, Function):
        yield
        return

    # 1. Filter: Only run for tests marked with @pytest.mark.assay
    marker = item.get_closest_marker("assay")
    if marker is None:
        yield
        return

    logger.info("Intercepting Agent.run() calls in pytest_runtest_call hook")

    # 2. Initialize Storage: Prepare the specific stash key for this test item
    item.stash[AGENT_RESPONSES_KEY] = []

    # 3. Capture State: Save the *current* method implementation.
    # Crucial for compatibility: If another plugin has already patched Agent.run,
    # we capture that patch instead of the raw method, preserving the chain of command.
    original_agent_run = Agent.run

    # 4. Set Context: Push the current test item into the thread-local context.
    # This 'token' is required to cleanly pop the context later.
    token = _current_item_var.set(item)

    # 5. Define Wrapper: Create the interceptor closure locally
    async def _instrumented_agent_run(
        self: Agent[Any, Any],
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> AgentRunResult[Any]:
        """Wrapped Agent.run() that captures responses to the current test item's stash."""
        # A. Execute actual logic (awaiting the captured variable avoids infinite recursion)
        result = await original_agent_run(self, *args, **kwargs)

        # B. Capture result (retrieve item via ContextVar tunnel)
        current_item = _current_item_var.get()
        if current_item is not None:
            responses = current_item.stash.get(AGENT_RESPONSES_KEY, [])
            responses.append(result)
            logger.debug(f"Captured Agent.run() response #{len(responses)}: {repr(result.output)[:100]}")

        return result

    # 6. Apply Patch: Hot-swap the class method
    Agent.run = _instrumented_agent_run  # type: ignore[method-assign]
    logger.debug("Monkeypatched Agent.run() for automatic response capture")

    try:
        yield  # 7. Yield control to pytest to run the actual test function
    finally:
        # 8. Data Integrity: Restore the original method immediately
        Agent.run = original_agent_run  # type: ignore[method-assign]

        # 9. Cleanup: Pop the ContextVar value using the token to prevent state leaks.
        # This ensures the context is clean for the next test execution.
        _current_item_var.reset(token)

        captured_count = len(item.stash.get(AGENT_RESPONSES_KEY, []))
        logger.debug(f"Restored Agent.run(), captured {captured_count} responses.")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item: Item, nextitem: Item | None) -> Generator[None, None, None]:
    """
    Hookwrapper for test teardown that runs evaluation or baseline serialization.

    Code before yield runs while other fixtures (including VCR cassette recording) are still
    active and after Agent.run has been restored by pytest_runtest_call.

    Args:
        item: The pytest test item being torn down.
        nextitem: The next test item (if any).
    """
    if _is_assay(item):
        assay: AssayContext | None = item.funcargs.get("context")  # type: ignore[attr-defined]
        if assay is not None:
            if assay.assay_mode == "new_baseline":
                _serialize_baseline(item, assay)
            elif assay.assay_mode == "evaluate":
                _run_evaluation(item, assay)

    yield  # Fixture finalization runs here (VCR cassette closes)
