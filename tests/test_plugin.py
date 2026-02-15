#!/usr/bin/env python3
"""
Unit tests for the pytest assay plugin.

These tests cover:
- Helper functions (_path, _is_assay)
- Plugin hook functions (pytest_addoption, pytest_configure, pytest_runtest_setup, etc.)
- Agent.run() interception mechanism (monkeypatch, capture, restoration)
- Evaluation and baseline serialization in pytest_runtest_teardown
"""

from __future__ import annotations as _annotations

import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_evals import Case, Dataset
from pytest import Function, Item

import pytest_assay.plugin
from pytest_assay.models import AssayContext, Readout
from pytest_assay.plugin import (
    AGENT_RESPONSES_KEY,
    BASELINE_DATASET_KEY,
    _is_assay,
    _path,
    pytest_addoption,
    pytest_configure,
    pytest_runtest_call,
    pytest_runtest_setup,
    pytest_runtest_teardown,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _drive_hookwrapper(item: Item, nextitem: Item | None = None) -> None:
    """Helper to drive pytest_runtest_teardown hookwrapper generator."""
    gen = pytest_runtest_teardown(item, nextitem)
    next(gen)  # Run pre-yield code (evaluation/serialization), then pause at yield
    with contextlib.suppress(StopIteration):
        next(gen)  # Resume after yield (fixture finalization)


# =============================================================================
# Helper Function Tests (_path, _is_assay)
# =============================================================================


def test_path_computation(mocker: MockerFixture) -> None:
    """Test _path computes the correct assay file path."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.path = Path("/project/tests/test_example.py")
    mock_item.name = "test_my_function"

    result = _path(mock_item)

    expected = Path("/project/tests/pytest_assay/test_example/test_my_function.json")
    assert result == expected

    # Verify path is absolute
    assert result.is_absolute()

    # Verify path ends with .json
    assert result.suffix == ".json"

    # Verify pytest_assay directory is in the path
    assert "pytest_assay" in result.parts


def test_path_computation_with_parametrized_test(mocker: MockerFixture) -> None:
    """Test _path strips parameter suffix from parametrized test names."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.path = Path("/project/tests/test_example.py")
    mock_item.name = "test_my_function[param1-param2]"

    result = _path(mock_item)

    # Should strip everything after the bracket
    expected = Path("/project/tests/pytest_assay/test_example/test_my_function.json")
    assert result == expected


def test_path_computation_nested_directory(mocker: MockerFixture) -> None:
    """Test _path with deeply nested test directories."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.path = Path("/project/tests/integration/api/test_endpoints.py")
    mock_item.name = "test_get_user"

    result = _path(mock_item)

    expected = Path("/project/tests/integration/api/pytest_assay/test_endpoints/test_get_user.json")
    assert result == expected


def test_is_assay_with_marker(mocker: MockerFixture) -> None:
    """Test _is_assay returns True for Function items with assay marker."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.get_closest_marker.return_value = mocker.MagicMock()

    assert _is_assay(mock_item) is True
    mock_item.get_closest_marker.assert_called_once_with("assay")


def test_is_assay_without_marker(mocker: MockerFixture) -> None:
    """Test _is_assay returns False for items without assay marker."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.get_closest_marker.return_value = None

    assert _is_assay(mock_item) is False


def test_is_assay_non_function_item(mocker: MockerFixture) -> None:
    """Test _is_assay returns False for non-Function items (e.g., class)."""
    mock_item = mocker.MagicMock(spec=Item)

    # Ensure it's not a Function instance
    assert not isinstance(mock_item, Function)
    assert _is_assay(mock_item) is False


# =============================================================================
# pytest_addoption Tests
# =============================================================================


def test_pytest_addoption(mocker: MockerFixture) -> None:
    """Test pytest_addoption registers the --assay-mode option correctly."""
    mock_parser = mocker.MagicMock()

    pytest_addoption(mock_parser)

    mock_parser.addoption.assert_called_once_with(
        "--assay-mode",
        action="store",
        default="evaluate",
        choices=("evaluate", "new_baseline"),
        help='Assay mode. Defaults to "evaluate".',
    )


# =============================================================================
# pytest_configure Tests
# =============================================================================


def test_pytest_configure(mocker: MockerFixture) -> None:
    """Test pytest_configure registers the assay marker."""
    mock_config = mocker.MagicMock()
    mocker.patch("pytest_assay.plugin.logger")

    pytest_configure(mock_config)

    mock_config.addinivalue_line.assert_called_once()
    call_args = mock_config.addinivalue_line.call_args
    assert call_args[0][0] == "markers"
    assert "assay" in call_args[0][1]


# =============================================================================
# pytest_runtest_setup Tests
# =============================================================================


def test_pytest_runtest_setup_non_assay_item(mocker: MockerFixture) -> None:
    """Test pytest_runtest_setup skips non-assay items."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.funcargs = {}
    mocker.patch("pytest_assay.plugin._is_assay", return_value=False)

    # Should not raise and should not modify funcargs
    pytest_runtest_setup(mock_item)

    # funcargs should not have been accessed/modified
    assert "context" not in mock_item.funcargs


def test_pytest_runtest_setup_with_existing_dataset(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup loads existing dataset from file."""
    # Create a temporary dataset file with realistic topic data
    dataset_path = tmp_path / "pytest_assay" / "test_module" / "test_func.json"
    dataset_path.parent.mkdir(parents=True)
    dataset = Dataset[dict[str, str], type[None], Any](
        cases=[
            Case(name="case_001", inputs={"topic": "pangolin trafficking", "query": "existing query"}),
            Case(name="case_002", inputs={"topic": "molecular gastronomy", "query": "another query"}),
        ]
    )
    dataset.to_file(dataset_path, schema_path=None)

    # Mock the item
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mocker.patch("pytest_assay.plugin._path", return_value=dataset_path)
    mocker.patch("pytest_assay.plugin.logger")

    pytest_runtest_setup(mock_item)

    # Verify baseline dataset was stashed for evaluators
    assert BASELINE_DATASET_KEY in mock_item.stash
    baseline = mock_item.stash[BASELINE_DATASET_KEY]
    assert len(baseline.cases) == 2
    assert baseline.cases[0].inputs["query"] == "existing query"

    # Verify assay context was injected
    assert "context" in mock_item.funcargs
    assay_ctx = mock_item.funcargs["context"]

    # Check by attribute presence instead of isinstance (module reload can cause different class instances)
    assert hasattr(assay_ctx, "dataset")
    assert hasattr(assay_ctx, "path")
    assert hasattr(assay_ctx, "assay_mode")

    # Verify dataset content was loaded correctly
    assert len(assay_ctx.dataset.cases) == 2
    assert assay_ctx.path == dataset_path
    assert assay_ctx.assay_mode == "evaluate"

    # Verify case data integrity
    assert assay_ctx.dataset.cases[0].name == "case_001"
    assert assay_ctx.dataset.cases[0].inputs["topic"] == "pangolin trafficking"
    assert assay_ctx.dataset.cases[0].inputs["query"] == "existing query"

    assert assay_ctx.dataset.cases[1].name == "case_002"
    assert assay_ctx.dataset.cases[1].inputs["topic"] == "molecular gastronomy"
    assert assay_ctx.dataset.cases[1].inputs["query"] == "another query"

    # Verify cases are iterable (as used in test_curiosity.py)
    topics = [case.inputs["topic"] for case in assay_ctx.dataset.cases]
    assert topics == ["pangolin trafficking", "molecular gastronomy"]


def test_pytest_runtest_setup_with_generator(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup calls generator when no dataset file exists."""
    dataset_path = tmp_path / "pytest_assay" / "test_module" / "test_func.json"

    # Mock generator that returns a dataset (like generate_evaluation_cases in test_curiosity.py)
    topics = ["pangolin trafficking", "molecular gastronomy", "dark kitchen economics"]
    generated_cases: list[Case[dict[str, str], type[None], Any]] = [
        Case(name=f"case_{idx:03d}", inputs={"topic": topic}) for idx, topic in enumerate(topics)
    ]
    generated_dataset = Dataset[dict[str, str], type[None], Any](cases=generated_cases)
    mock_generator = mocker.MagicMock(return_value=generated_dataset)

    # Mock the item
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"generator": mock_generator}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mocker.patch("pytest_assay.plugin._path", return_value=dataset_path)
    mocker.patch("pytest_assay.plugin.logger")

    pytest_runtest_setup(mock_item)

    # Verify baseline dataset was stashed
    assert BASELINE_DATASET_KEY in mock_item.stash
    assert len(mock_item.stash[BASELINE_DATASET_KEY].cases) == 3

    # Verify generator was called exactly once
    mock_generator.assert_called_once()

    # Verify dataset file was created
    assert dataset_path.exists()
    assert dataset_path.suffix == ".json"

    # Verify file contains valid JSON that can be reloaded
    reloaded_dataset = Dataset[dict[str, str], type[None], Any].from_file(dataset_path)
    assert len(reloaded_dataset.cases) == 3

    # Verify reloaded data matches original
    for idx, topic in enumerate(topics):
        assert reloaded_dataset.cases[idx].name == f"case_{idx:03d}"
        assert reloaded_dataset.cases[idx].inputs["topic"] == topic

    # Verify assay context was injected with correct data
    assay_ctx = mock_item.funcargs["context"]
    assert len(assay_ctx.dataset.cases) == 3
    assert assay_ctx.path == dataset_path
    assert assay_ctx.assay_mode == "evaluate"

    # Verify case content
    assert assay_ctx.dataset.cases[0].inputs["topic"] == "pangolin trafficking"
    assert assay_ctx.dataset.cases[1].inputs["topic"] == "molecular gastronomy"
    assert assay_ctx.dataset.cases[2].inputs["topic"] == "dark kitchen economics"


def test_pytest_runtest_setup_generator_invalid_return(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup raises TypeError for invalid generator return."""
    dataset_path = tmp_path / "pytest_assay" / "test_module" / "test_func.json"

    # Mock generator that returns invalid type
    mock_generator = mocker.MagicMock(return_value="not a dataset")

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"generator": mock_generator}
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mocker.patch("pytest_assay.plugin._path", return_value=dataset_path)
    mocker.patch("pytest_assay.plugin.logger")

    with pytest.raises(TypeError, match="must return a Dataset instance"):
        pytest_runtest_setup(mock_item)


def test_pytest_runtest_setup_empty_dataset(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup uses empty dataset when no file or generator."""
    dataset_path = tmp_path / "pytest_assay" / "test_module" / "test_func.json"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}  # No generator
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mocker.patch("pytest_assay.plugin._path", return_value=dataset_path)
    mocker.patch("pytest_assay.plugin.logger")

    pytest_runtest_setup(mock_item)

    assert BASELINE_DATASET_KEY in mock_item.stash
    assert len(mock_item.stash[BASELINE_DATASET_KEY].cases) == 0

    assay_ctx = mock_item.funcargs["context"]
    assert len(assay_ctx.dataset.cases) == 0


def test_pytest_runtest_setup_baseline_stash_is_copy(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test that mutating assay.dataset does not change the stashed baseline."""
    dataset_path = tmp_path / "pytest_assay" / "test_module" / "test_func.json"
    dataset_path.parent.mkdir(parents=True)
    dataset = Dataset[dict[str, str], type[None], Any](
        cases=[
            Case(name="case_001", inputs={"topic": "topic A", "query": "baseline query A"}),
        ]
    )
    dataset.to_file(dataset_path, schema_path=None)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mocker.patch("pytest_assay.plugin._path", return_value=dataset_path)
    mocker.patch("pytest_assay.plugin.logger")

    pytest_runtest_setup(mock_item)

    assay_ctx = mock_item.funcargs["context"]
    baseline = mock_item.stash[BASELINE_DATASET_KEY]

    # Simulate test mutation (like test_curiosity.py)
    assay_ctx.dataset.cases.clear()
    assay_ctx.dataset.cases.append(Case(name="case_001", inputs={"topic": "topic A", "query": "novel query A"}))

    # Stashed baseline must be unchanged
    assert len(baseline.cases) == 1
    assert baseline.cases[0].inputs["query"] == "baseline query A"
    assert assay_ctx.dataset.cases[0].inputs["query"] == "novel query A"


# =============================================================================
# pytest_runtest_call Tests
# =============================================================================


def test_pytest_runtest_call_non_function_item(mocker: MockerFixture) -> None:
    """Test pytest_runtest_call yields immediately for non-Function items."""
    mock_item = mocker.MagicMock(spec=Item)

    gen = pytest_runtest_call(mock_item)
    next(gen)  # Should yield immediately

    with pytest.raises(StopIteration):
        next(gen)


def test_pytest_runtest_call_without_assay_marker(mocker: MockerFixture) -> None:
    """Test pytest_runtest_call yields immediately without assay marker."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.get_closest_marker.return_value = None

    gen = pytest_runtest_call(mock_item)
    next(gen)

    with pytest.raises(StopIteration):
        next(gen)


def test_pytest_runtest_call_initializes_stash(mocker: MockerFixture) -> None:
    """Test pytest_runtest_call initializes the response stash with AGENT_RESPONSES_KEY."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("pytest_assay.plugin.logger")

    gen = pytest_runtest_call(mock_item)
    next(gen)  # Run until yield

    # Verify stash was initialized with the correct key
    assert AGENT_RESPONSES_KEY in mock_item.stash
    assert mock_item.stash[AGENT_RESPONSES_KEY] == []

    # Clean up
    with contextlib.suppress(StopIteration):
        next(gen)


def test_pytest_runtest_call_sets_context_var(mocker: MockerFixture) -> None:
    """Test pytest_runtest_call sets the current item context variable."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("pytest_assay.plugin.logger")

    gen = pytest_runtest_call(mock_item)
    next(gen)

    # During the yield, context var should be set
    assert pytest_assay.plugin._current_item_var.get() == mock_item

    # Clean up and verify context var is reset
    with contextlib.suppress(StopIteration):
        next(gen)

    assert pytest_assay.plugin._current_item_var.get() is None


def test_pytest_runtest_call_restores_agent_run(mocker: MockerFixture) -> None:
    """Test that Agent.run is restored to its original method after the hook completes."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("pytest_assay.plugin.logger")

    original_run = Agent.run

    gen = pytest_runtest_call(mock_item)
    next(gen)  # Run until yield; Agent.run is now monkeypatched

    # During the yield, Agent.run should be the instrumented version
    assert Agent.run is not original_run

    # Clean up
    with contextlib.suppress(StopIteration):
        next(gen)

    # After the hook, Agent.run must be restored
    assert Agent.run is original_run


@pytest.mark.asyncio
async def test_instrumented_agent_run_captures_response(mocker: MockerFixture) -> None:
    """
    Test that calling Agent.run() while the monkeypatch is active captures the response.

    Verifies:
    - The original Agent.run is called (no infinite recursion)
    - The result is appended to item.stash[AGENT_RESPONSES_KEY]
    - The return value is passed through to the caller
    """
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("pytest_assay.plugin.logger")

    # Mock the original Agent.run to return a controlled result
    mock_result = mocker.MagicMock(spec=AgentRunResult)
    mock_result.output = "captured output"
    mock_original_run = AsyncMock(return_value=mock_result)
    mocker.patch.object(Agent, "run", mock_original_run)

    gen = pytest_runtest_call(mock_item)
    next(gen)  # Activates the monkeypatch

    # Call the instrumented Agent.run
    agent = mocker.MagicMock(spec=Agent)
    result = await Agent.run(agent, user_prompt="test prompt")

    # Return value should be passed through
    assert result is mock_result

    # Response should be captured in the stash
    assert len(mock_item.stash[AGENT_RESPONSES_KEY]) == 1
    assert mock_item.stash[AGENT_RESPONSES_KEY][0] is mock_result

    # Original Agent.run should have been called
    mock_original_run.assert_called_once()

    # Clean up
    with contextlib.suppress(StopIteration):
        next(gen)


@pytest.mark.asyncio
async def test_instrumented_agent_run_no_item_in_context(mocker: MockerFixture) -> None:
    """
    Test that _instrumented_agent_run works when _current_item_var is None.

    When Agent.run() is called outside a test body (context var unset), the instrumented
    function should call the original and return the result without appending to any stash.
    """
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("pytest_assay.plugin.logger")

    # Mock the original Agent.run
    mock_result = mocker.MagicMock(spec=AgentRunResult)
    mock_result.output = "passthrough output"
    mock_original_run = AsyncMock(return_value=mock_result)
    mocker.patch.object(Agent, "run", mock_original_run)

    gen = pytest_runtest_call(mock_item)
    next(gen)  # Activates the monkeypatch; _current_item_var is set to mock_item

    # Manually clear the context var to simulate an external call
    token = pytest_assay.plugin._current_item_var.set(None)

    try:
        agent = mocker.MagicMock(spec=Agent)
        result = await Agent.run(agent, user_prompt="external call")

        # Return value should still pass through
        assert result is mock_result

        # No response should be captured (context var was None)
        assert mock_item.stash[AGENT_RESPONSES_KEY] == []
    finally:
        # Restore the context var so the generator cleanup works correctly
        pytest_assay.plugin._current_item_var.reset(token)

    # Clean up
    with contextlib.suppress(StopIteration):
        next(gen)


# =============================================================================
# pytest_runtest_teardown Tests (_serialize_baseline path)
# =============================================================================


def test_pytest_runtest_teardown_non_assay_item(mocker: MockerFixture) -> None:
    """Test pytest_runtest_teardown skips non-assay items."""
    mock_item = mocker.MagicMock(spec=Item)
    mocker.patch("pytest_assay.plugin._is_assay", return_value=False)

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)


def test_pytest_runtest_teardown_no_assay_context(mocker: MockerFixture) -> None:
    """Test pytest_runtest_teardown handles missing assay context gracefully."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}  # No assay context

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)

    # Drive the hookwrapper generator - should not raise
    _drive_hookwrapper(mock_item, None)


def test_pytest_runtest_teardown_new_baseline_mode(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown merges captured responses and serializes dataset in new_baseline mode."""
    dataset_path = tmp_path / "pytest_assay" / "test.json"

    # Create dataset with cases that have empty expected_output (as generated)
    cases: list[Case[dict[str, str], str, Any]] = [
        Case(name="case_000", inputs={"topic": "topic A"}, expected_output=""),
        Case(name="case_001", inputs={"topic": "topic B"}, expected_output=""),
    ]
    dataset = Dataset[dict[str, str], str, Any](cases=cases)

    # Create mock AgentRunResult responses (captured by pytest_runtest_call)
    mock_response_a = mocker.MagicMock(spec=AgentRunResult)
    mock_response_a.output = "generated query A"
    mock_response_b = mocker.MagicMock(spec=AgentRunResult)
    mock_response_b.output = "generated query B"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="new_baseline")}
    mock_item.stash = {
        AGENT_RESPONSES_KEY: [mock_response_a, mock_response_b],
    }

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mocker.patch("pytest_assay.plugin.logger")

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # File should be created in new_baseline mode
    assert dataset_path.exists()
    assert dataset_path.suffix == ".json"

    # Verify serialized content can be reloaded
    reloaded = Dataset[dict[str, str], str, Any].from_file(dataset_path)
    assert len(reloaded.cases) == 2

    # Verify expected_output was populated from captured responses
    assert reloaded.cases[0].name == "case_000"
    assert reloaded.cases[0].inputs["topic"] == "topic A"
    assert reloaded.cases[0].expected_output == "generated query A"

    assert reloaded.cases[1].name == "case_001"
    assert reloaded.cases[1].inputs["topic"] == "topic B"
    assert reloaded.cases[1].expected_output == "generated query B"


def test_pytest_runtest_teardown_new_baseline_response_count_mismatch(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown skips serialization when response count mismatches case count."""
    dataset_path = tmp_path / "pytest_assay" / "test.json"

    cases: list[Case[dict[str, str], str, Any]] = [
        Case(name="case_000", inputs={"topic": "topic A"}, expected_output=""),
        Case(name="case_001", inputs={"topic": "topic B"}, expected_output=""),
    ]
    dataset = Dataset[dict[str, str], str, Any](cases=cases)

    # Only one response for two cases
    mock_response = mocker.MagicMock(spec=AgentRunResult)
    mock_response.output = "query A"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="new_baseline")}
    mock_item.stash = {
        AGENT_RESPONSES_KEY: [mock_response],
    }

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("pytest_assay.plugin.logger")

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # File should NOT be created due to mismatch
    assert not dataset_path.exists()
    mock_logger.error.assert_called_once()
    assert "Cannot merge responses" in str(mock_logger.error.call_args)


def test_pytest_runtest_teardown_new_baseline_none_output(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown uses empty string for None response output."""
    dataset_path = tmp_path / "pytest_assay" / "test.json"

    cases: list[Case[dict[str, str], str, Any]] = [
        Case(name="case_000", inputs={"topic": "topic A"}, expected_output=""),
    ]
    dataset = Dataset[dict[str, str], str, Any](cases=cases)

    mock_response = mocker.MagicMock(spec=AgentRunResult)
    mock_response.output = None

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="new_baseline")}
    mock_item.stash = {
        AGENT_RESPONSES_KEY: [mock_response],
    }

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mocker.patch("pytest_assay.plugin.logger")

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    assert dataset_path.exists()
    reloaded = Dataset[dict[str, str], str, Any].from_file(dataset_path)
    assert reloaded.cases[0].expected_output == ""


def test_pytest_runtest_teardown_new_baseline_no_responses(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown skips serialization when no responses captured but cases exist."""
    dataset_path = tmp_path / "pytest_assay" / "test.json"

    cases: list[Case[dict[str, str], str, Any]] = [
        Case(name="case_000", inputs={"topic": "topic A"}, expected_output=""),
    ]
    dataset = Dataset[dict[str, str], str, Any](cases=cases)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="new_baseline")}
    mock_item.stash = {}  # No AGENT_RESPONSES_KEY

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("pytest_assay.plugin.logger")

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # File should NOT be created (0 responses vs 1 case)
    assert not dataset_path.exists()
    mock_logger.error.assert_called_once()


# =============================================================================
# pytest_runtest_teardown Tests (_run_evaluation path)
# =============================================================================


def test_pytest_runtest_teardown_evaluate_mode(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown does not serialize dataset in evaluate mode."""
    dataset_path = tmp_path / "pytest_assay" / "test.json"
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="evaluate")}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"evaluator": AsyncMock(return_value=Readout(passed=True))}
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mocker.patch("pytest_assay.plugin.logger")
    mocker.patch("pytest_assay.plugin.asyncio.run")  # Mock to prevent actual evaluation

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # File should not be created in evaluate mode (evaluation doesn't serialize dataset)
    assert not dataset_path.exists()


def test_pytest_runtest_teardown_runs_evaluation(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown runs evaluation and serializes readout in evaluate mode."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])
    assay_path = tmp_path / "pytest_assay" / "test_module" / "test_func.json"
    assay_path.parent.mkdir(parents=True, exist_ok=True)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=assay_path, assay_mode="evaluate")}
    mock_marker = mocker.MagicMock()

    # Custom evaluator mock - returns Readout with details
    mock_evaluator = AsyncMock(return_value=Readout(passed=True, details={"test_key": "test_value"}))
    mock_marker.kwargs = {"evaluator": mock_evaluator}
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("pytest_assay.plugin.logger")

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # Evaluator should have been called
    mock_evaluator.assert_called_once_with(mock_item)
    # Should log the result
    mock_logger.info.assert_any_call("Evaluation result: passed=True")

    # Verify readout file was created
    readout_path = assay_path.with_suffix(".readout.json")
    assert readout_path.exists()

    # Verify readout content
    with readout_path.open() as f:
        readout_data = json.load(f)
    assert readout_data["passed"] is True
    assert readout_data["details"] == {"test_key": "test_value"}


def test_pytest_runtest_teardown_uses_default_evaluator(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown uses BradleyTerryEvaluator() as default."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])
    assay_path = tmp_path / "pytest_assay" / "test_module" / "test_func.json"
    assay_path.parent.mkdir(parents=True, exist_ok=True)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=assay_path, assay_mode="evaluate")}
    mock_item.stash = {
        AGENT_RESPONSES_KEY: [],
        BASELINE_DATASET_KEY: dataset,
    }
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}  # No evaluator specified - should use default
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("pytest_assay.plugin.logger")

    # Mock BradleyTerryEvaluator so the default evaluator is a mock we control
    mock_bt_evaluator = AsyncMock(return_value=Readout(passed=True, details={"default": True}))
    mocker.patch("pytest_assay.plugin.BradleyTerryEvaluator", return_value=mock_bt_evaluator)

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # The default BradleyTerryEvaluator instance should have been called
    mock_bt_evaluator.assert_called_once_with(mock_item)
    # Should log evaluation completed (no error)
    assert any("Evaluation result" in str(call) for call in mock_logger.info.call_args_list)


def test_pytest_runtest_teardown_invalid_evaluator(mocker: MockerFixture) -> None:
    """Test pytest_runtest_teardown handles non-callable evaluator."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"evaluator": "not_callable"}  # Invalid
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("pytest_assay.plugin.logger")

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # Should log error for invalid evaluator
    mock_logger.error.assert_called_once()
    assert "Invalid evaluator type" in str(mock_logger.error.call_args)


def test_pytest_runtest_teardown_evaluation_exception(mocker: MockerFixture) -> None:
    """Test pytest_runtest_teardown handles evaluation exceptions."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_marker = mocker.MagicMock()

    # Evaluator that raises an exception
    async def failing_evaluator(item: Item) -> Readout:
        raise RuntimeError("Evaluation failed")

    mock_marker.kwargs = {"evaluator": failing_evaluator}
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("pytest_assay.plugin.logger")

    # Drive the hookwrapper generator - should not raise, exception is handled internally
    _drive_hookwrapper(mock_item, None)

    # Should log exception
    mock_logger.exception.assert_called_once()


# =============================================================================
# Workflow Tests (setup + teardown combined)
# =============================================================================


def test_full_assay_workflow_with_topic_generation(mocker: MockerFixture, tmp_path: Path) -> None:
    """
    Test a realistic assay workflow similar to test_curiosity.py.

    This test verifies the complete flow:
    1. Generator creates initial cases with topics
    2. Setup injects AssayContext
    3. Agent.run() responses are captured in AGENT_RESPONSES_KEY
    4. Teardown merges responses into expected_output and serializes
    """
    dataset_path = tmp_path / "pytest_assay" / "test_curiosity" / "test_search_queries.json"

    # Define topics like in test_curiosity.py
    topics = ["pangolin trafficking networks", "molecular gastronomy", "dark kitchen economics"]

    # Generator function (similar to generate_evaluation_cases)
    def generate_cases() -> Dataset[dict[str, str], str, Any]:
        cases: list[Case[dict[str, str], str, Any]] = []
        for idx, topic in enumerate(topics):
            case = Case(name=f"case_{idx:03d}", inputs={"topic": topic}, expected_output="")
            cases.append(case)
        return Dataset[dict[str, str], str, Any](cases=cases)

    # Setup: Create mock item with generator
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"generator": generate_cases}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "new_baseline"

    mocker.patch("pytest_assay.plugin._is_assay", return_value=True)
    mocker.patch("pytest_assay.plugin._path", return_value=dataset_path)
    mocker.patch("pytest_assay.plugin.logger")

    # Run setup
    pytest_runtest_setup(mock_item)

    # Verify setup injected correct context
    assert "context" in mock_item.funcargs
    assay_ctx = mock_item.funcargs["context"]
    assert len(assay_ctx.dataset.cases) == 3
    assert assay_ctx.assay_mode == "new_baseline"

    # Simulate captured Agent.run() responses (populated by pytest_runtest_call)
    mock_responses = []
    for topic in topics:
        mock_response = mocker.MagicMock(spec=AgentRunResult)
        mock_response.output = f"search for: {topic}"
        mock_responses.append(mock_response)
    mock_item.stash[AGENT_RESPONSES_KEY] = mock_responses

    # Run teardown (should merge responses and serialize in new_baseline mode)
    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # Verify file was created with updated data
    assert dataset_path.exists()

    # Reload and verify data integrity
    reloaded = Dataset[dict[str, str], str, Any].from_file(dataset_path)
    assert len(reloaded.cases) == 3

    for idx, topic in enumerate(topics):
        assert reloaded.cases[idx].name == f"case_{idx:03d}"
        assert reloaded.cases[idx].inputs["topic"] == topic
        assert reloaded.cases[idx].expected_output == f"search for: {topic}"
