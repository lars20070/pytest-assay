#!/usr/bin/env python3
"""Unit tests for the PairwiseEvaluator."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Case, Dataset
from pytest import Function

import pytest_assay.plugin
from pytest_assay.evaluators.pairwise import EVALUATION_INSTRUCTIONS, PairwiseEvaluator
from pytest_assay.models import AssayContext, Readout
from pytest_assay.plugin import AGENT_RESPONSES_KEY, BASELINE_DATASET_KEY

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# ---------------------------------------------------------------------------
# Unit tests for PairwiseEvaluator
# ---------------------------------------------------------------------------


class TestPairwiseEvaluator:
    """Unit tests for PairwiseEvaluator."""

    def test_init_defaults(self, ollama_model: str) -> None:
        """Test PairwiseEvaluator initializes with default values."""
        evaluator = PairwiseEvaluator()

        # Default model: OpenAIChatModel on Ollama
        assert evaluator.model is not None
        assert isinstance(evaluator.model, OpenAIChatModel)
        assert evaluator.model.model_name == ollama_model
        # model_settings (TypedDict)
        assert evaluator.model_settings.get("temperature") == 0.0
        assert evaluator.model_settings.get("timeout") == 300
        # system_prompt
        assert evaluator.system_prompt is not None
        assert evaluator.system_prompt == EVALUATION_INSTRUCTIONS
        # agent
        assert evaluator.agent is not None
        # criterion
        assert evaluator.criterion == "Which of the two responses is better?"

    def test_init_custom(self) -> None:
        """Test PairwiseEvaluator initializes with custom criterion."""
        evaluator = PairwiseEvaluator(criterion="Custom criterion")

        assert evaluator.criterion == "Custom criterion"

    def test_init_with_custom_model(self) -> None:
        """Test PairwiseEvaluator uses custom model when provided."""
        custom_model = OpenAIChatModel(
            model_name="custom-model",
            provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        )
        evaluator = PairwiseEvaluator(model=custom_model)

        assert evaluator.model is custom_model
        assert isinstance(evaluator.model, OpenAIChatModel)
        assert evaluator.model.model_name == "custom-model"
        assert evaluator.agent.model is custom_model

    def test_init_with_model_string(self, mocker: MockerFixture) -> None:
        """Test PairwiseEvaluator accepts model as string (e.g. for OpenAI default)."""
        mocker.patch("pytest_assay.evaluators.pairwise.Agent")  # Mock Agent to avoid actual initialization and API keys
        evaluator = PairwiseEvaluator(model="openai:gpt-4o-mini")

        assert evaluator.model == "openai:gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_call_no_pairs(self, mocker: MockerFixture) -> None:
        """Test __call__ with empty baseline and novel lists returns passed=False."""
        evaluator = PairwiseEvaluator()
        mock_item = mocker.MagicMock(spec=Function)
        mock_item.funcargs = {
            "context": AssayContext(
                dataset=Dataset[dict[str, str], type[None], Any](cases=[]),
                path=Path("/tmp/test.json"),
                assay_mode="evaluate",
            )
        }
        mock_item.stash = {
            AGENT_RESPONSES_KEY: [],
            BASELINE_DATASET_KEY: Dataset[dict[str, str], type[None], Any](cases=[]),
        }

        mocker.patch("pytest_assay.evaluators.pairwise.logger")

        result = await evaluator(mock_item)

        assert isinstance(result, Readout)
        assert result.passed is False
        assert result.details is not None
        assert result.details.get("test_cases_count") == 0
        assert result.details.get("wins_baseline") == []
        assert result.details.get("wins_novel") == []

    @pytest.mark.asyncio
    async def test_call_novel_wins(self, mocker: MockerFixture) -> None:
        """Test __call__ when novel wins all comparisons."""
        evaluator = PairwiseEvaluator(criterion="Test criterion")

        cases: list[Case[dict[str, str], type[None], Any]] = [
            Case(name="case_001", inputs={"query": "baseline query 1"}),
            Case(name="case_002", inputs={"query": "baseline query 2"}),
        ]
        dataset = Dataset[dict[str, str], type[None], Any](cases=cases)

        mock_response1 = mocker.MagicMock(spec=AgentRunResult)
        mock_response1.output = "novel output 1"
        mock_response2 = mocker.MagicMock(spec=AgentRunResult)
        mock_response2.output = "novel output 2"

        mock_item = mocker.MagicMock(spec=Function)
        mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
        mock_item.stash = {
            AGENT_RESPONSES_KEY: [mock_response1, mock_response2],
            pytest_assay.plugin.BASELINE_DATASET_KEY: dataset,
        }

        mocker.patch("pytest_assay.evaluators.pairwise.logger")

        # Mock agent.run to return "B" (novel wins) for both comparisons
        mock_run_result = mocker.MagicMock()
        mock_run_result.output = "B"
        mocker.patch.object(evaluator.agent, "run", new_callable=AsyncMock, return_value=mock_run_result)

        result = await evaluator(mock_item)

        assert isinstance(result, Readout)
        assert result.passed is True
        assert result.details is not None
        assert result.details.get("test_cases_count") == 2
        assert result.details.get("wins_novel") == [True, True]
        assert result.details.get("wins_baseline") == [False, False]

    @pytest.mark.asyncio
    async def test_call_baseline_wins(self, mocker: MockerFixture) -> None:
        """Test __call__ when baseline wins more comparisons."""
        evaluator = PairwiseEvaluator(criterion="Test criterion")

        cases: list[Case[dict[str, str], type[None], Any]] = [
            Case(name="case_001", inputs={"query": "baseline query"}),
        ]
        dataset = Dataset[dict[str, str], type[None], Any](cases=cases)

        mock_response = mocker.MagicMock(spec=AgentRunResult)
        mock_response.output = "novel output"

        mock_item = mocker.MagicMock(spec=Function)
        mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
        mock_item.stash = {
            AGENT_RESPONSES_KEY: [mock_response],
            pytest_assay.plugin.BASELINE_DATASET_KEY: dataset,
        }

        mocker.patch("pytest_assay.evaluators.pairwise.logger")

        # Mock agent.run to return "A" (baseline wins)
        mock_run_result = mocker.MagicMock()
        mock_run_result.output = "A"
        mocker.patch.object(evaluator.agent, "run", new_callable=AsyncMock, return_value=mock_run_result)

        result = await evaluator(mock_item)

        assert isinstance(result, Readout)
        assert result.passed is False
        assert result.details is not None
        assert result.details.get("wins_novel") == [False]
        assert result.details.get("wins_baseline") == [True]

    @pytest.mark.asyncio
    async def test_call_tie_fails(self, mocker: MockerFixture) -> None:
        """Test __call__ when wins are tied (novel must strictly exceed baseline to pass)."""
        evaluator = PairwiseEvaluator()

        cases: list[Case[dict[str, str], type[None], Any]] = [
            Case(name="case_001", inputs={"query": "q1"}),
            Case(name="case_002", inputs={"query": "q2"}),
        ]
        dataset = Dataset[dict[str, str], type[None], Any](cases=cases)

        mock_response1 = mocker.MagicMock(spec=AgentRunResult)
        mock_response1.output = "novel 1"
        mock_response2 = mocker.MagicMock(spec=AgentRunResult)
        mock_response2.output = "novel 2"

        mock_item = mocker.MagicMock(spec=Function)
        mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
        mock_item.stash = {
            AGENT_RESPONSES_KEY: [mock_response1, mock_response2],
            pytest_assay.plugin.BASELINE_DATASET_KEY: dataset,
        }

        mocker.patch("pytest_assay.evaluators.pairwise.logger")

        # First call returns "B" (novel wins), second returns "A" (baseline wins) → tie
        mock_result_b = mocker.MagicMock()
        mock_result_b.output = "B"
        mock_result_a = mocker.MagicMock()
        mock_result_a.output = "A"
        mocker.patch.object(evaluator.agent, "run", new_callable=AsyncMock, side_effect=[mock_result_b, mock_result_a])

        result = await evaluator(mock_item)

        assert isinstance(result, Readout)
        assert result.passed is False
        assert result.details is not None
        assert result.details.get("wins_novel") == [True, False]
        assert result.details.get("wins_baseline") == [False, True]

    @pytest.mark.asyncio
    async def test_call_mismatch_raises(self, mocker: MockerFixture) -> None:
        """Test __call__ raises when baseline and novel counts differ."""
        evaluator = PairwiseEvaluator()
        cases: list[Case[dict[str, str], type[None], Any]] = [
            Case(name="case_001", inputs={"query": "baseline query"}),
        ]
        dataset = Dataset[dict[str, str], type[None], Any](cases=cases)

        mock_item = mocker.MagicMock(spec=Function)
        mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
        mock_item.stash = {
            AGENT_RESPONSES_KEY: [],  # No novel responses
            pytest_assay.plugin.BASELINE_DATASET_KEY: dataset,
        }

        mocker.patch("pytest_assay.evaluators.pairwise.logger")

        with pytest.raises(AssertionError, match="Mismatch in response counts"):
            await evaluator(mock_item)

    @pytest.mark.asyncio
    async def test_call_skips_none_output(self, mocker: MockerFixture) -> None:
        """Test __call__ skips novel responses with None output."""
        evaluator = PairwiseEvaluator()

        # No baseline cases → 0 baseline responses
        dataset = Dataset[dict[str, str], type[None], Any](cases=[])

        mock_response = mocker.MagicMock(spec=AgentRunResult)
        mock_response.output = None  # None output should be skipped

        mock_item = mocker.MagicMock(spec=Function)
        mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
        mock_item.stash = {
            AGENT_RESPONSES_KEY: [mock_response],
            BASELINE_DATASET_KEY: dataset,
        }

        mocker.patch("pytest_assay.evaluators.pairwise.logger")

        # 0 baseline, 0 novel (the None was skipped) → no mismatch, passed=False
        result = await evaluator(mock_item)

        assert isinstance(result, Readout)
        assert result.passed is False
        assert result.details is not None
        assert result.details.get("test_cases_count") == 0

    @pytest.mark.asyncio
    async def test_call_prompt_contains_criterion_and_responses(self, mocker: MockerFixture) -> None:
        """Test the prompt passed to the agent includes criterion, baseline, and novel text."""
        evaluator = PairwiseEvaluator(criterion="creativity")

        cases: list[Case[dict[str, str], str, Any]] = [
            Case(name="case_001", inputs={"query": "q1"}, expected_output="vanilla"),
        ]
        dataset = Dataset[dict[str, str], str, Any](cases=cases)

        mock_response = mocker.MagicMock(spec=AgentRunResult)
        mock_response.output = "miso caramel"

        mock_item = mocker.MagicMock(spec=Function)
        mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
        mock_item.stash = {
            AGENT_RESPONSES_KEY: [mock_response],
            BASELINE_DATASET_KEY: dataset,
        }

        mocker.patch("pytest_assay.evaluators.pairwise.logger")

        mock_run_result = mocker.MagicMock()
        mock_run_result.output = "A"
        mock_run = mocker.patch.object(evaluator.agent, "run", new_callable=AsyncMock, return_value=mock_run_result)

        await evaluator(mock_item)

        call_kwargs = mock_run.call_args.kwargs
        prompt = call_kwargs["user_prompt"]
        assert "creativity" in prompt
        assert "vanilla" in prompt  # baseline from expected_output
        assert "miso caramel" in prompt  # novel from agent response

    @pytest.mark.asyncio
    async def test_protocol_conformance(self) -> None:
        """Test PairwiseEvaluator conforms to Evaluator Protocol."""
        evaluator = PairwiseEvaluator()
        assert callable(evaluator)
        assert inspect.iscoroutinefunction(evaluator.__call__)
