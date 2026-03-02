#!/usr/bin/env python3
"""
Integration tests for the assay pytest plugin with Ollama.

These tests require a local Ollama instance and are marked with @pytest.mark.ollama.
They generate search queries for research topics and evaluate them using different strategies:
1. PairwiseEvaluator: Compares baseline vs novel responses directly
2. BradleyTerryEvaluator: Ranks all responses using the Bradley-Terry model
3. LengthEvaluator: Checks whether novel responses are longer than baseline responses. LengthEvaluator is a custom user-defined evaluator and
   not part of the pytest-assay package, demonstrating how users can implement their own evaluation logic.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

import pytest
from _ollama import OLLAMA_BASE_URL, OLLAMA_MODEL
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Case, Dataset

from pytest_assay.logger import logger
from pytest_assay.models import EvaluatorInput, Readout
from pytest_assay.plugin import BradleyTerryEvaluator, PairwiseEvaluator

if TYPE_CHECKING:
    from pydantic_ai.settings import ModelSettings

    from pytest_assay.models import AssayContext


def generate_evaluation_cases() -> Dataset[dict[str, str], str, Any]:
    """Generate a list of Cases containing topics as input."""
    logger.info("Creating new assay dataset.")

    topics = [
        "pangolin trafficking networks",
        "molecular gastronomy",
        "dark kitchen economics",
        "kintsugi philosophy",
        "nano-medicine delivery systems",
        "Streisand effect dynamics",
        "Anne Brorhilker",
        "bioconcrete self-healing",
        "bacteriophage therapy revival",
        "Habsburg jaw genetics",
    ]

    cases: list[Case[dict[str, str], str, Any]] = []
    for idx, topic in enumerate(topics):
        logger.debug(f"Case {idx + 1} / {len(topics)} with topic: {topic}")
        case = Case(
            name=f"case_{idx:03d}",
            inputs={"topic": topic},
            expected_output="",
        )
        cases.append(case)

    return Dataset[dict[str, str], str, Any](cases=cases)


model = OpenAIChatModel(
    model_name=OLLAMA_MODEL,
    provider=OpenAIProvider(base_url=f"{OLLAMA_BASE_URL}/v1"),
)

BASIC_PROMPT = "Please generate a useful search query for the following research topic: <TOPIC>{topic}</TOPIC>"

CREATIVE_PROMPT = """
Please generate a very creative search query for the research topic: <TOPIC>{topic}</TOPIC>
The query should show genuine originality and interest in the topic. AVOID any generic or formulaic phrases.

Examples of formulaic queries for the topic 'molecular gastronomy' (BAD):
- 'Definition molecular gastronomy'
- 'Molecular gastronomy techniques and applications'

Examples of curious, creative queries for the topic 'molecular gastronomy' (GOOD):
- 'Use of liquid nitrogen instead of traditional freezing for food texture'
- 'Failed molecular gastronomy experiments that led to new dishes'

Now generate one creative search query for: <TOPIC>{topic}</TOPIC>
"""


async def _run_query_generation(context: AssayContext, model_settings: ModelSettings) -> None:
    """Run query generation for all cases in the dataset."""
    query_agent = Agent(
        model=model,
        output_type=str,
        system_prompt="Please generate a concise web search query for the given research topic. You must respond only in English.",
        retries=5,
        instrument=True,
    )

    for case in context.dataset.cases:
        logger.info(f"Case {case.name} with topic: {case.inputs['topic']}")

        # prompt = BASIC_PROMPT.format(topic=case.inputs["topic"])  # for assay-mode 'new_baseline'
        prompt = CREATIVE_PROMPT.format(topic=case.inputs["topic"])  # for assay-mode 'evaluate'

        async with query_agent:
            result = await query_agent.run(
                user_prompt=prompt,
                model_settings=model_settings,
            )

        logger.debug(f"Generated search query: {result.output}")


@pytest.mark.ollama
@pytest.mark.assay(
    generator=generate_evaluation_cases,
    evaluator=PairwiseEvaluator(
        model=model,
        criterion="Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?",
    ),
)
@pytest.mark.asyncio
async def test_integration_pairwiseevaluator(context: AssayContext, model_settings: ModelSettings) -> None:
    """
    Integration test for the assay pytest plugin with PairwiseEvaluator evaluator.

    An agent generates search queries for various research topics.
    PairwiseEvaluator then evaluates the creativity of the generated queries.

    Args:
        context: The assay context containing the evaluation dataset context.dataset and other information.
        model_settings: Deterministic model settings for reproducible tests.
    """
    logger.info("Integration test for assay pytest plugin with PairwiseEvaluator.")
    await _run_query_generation(context, model_settings)


@pytest.mark.ollama
@pytest.mark.assay(
    generator=generate_evaluation_cases,
    evaluator=BradleyTerryEvaluator(
        model=model,
        criterion="Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?",
        max_standard_deviation=2.1,
    ),
)
@pytest.mark.asyncio
async def test_integration_bradleyterryevaluator(context: AssayContext, model_settings: ModelSettings) -> None:
    """
    Integration test for the assay pytest plugin with BradleyTerryEvaluator evaluator.

    An agent generates search queries for various research topics.
    BradleyTerryEvaluator then evaluates the creativity of the generated queries.

    Args:
        context: The assay context containing the evaluation dataset context.dataset and other information.
        model_settings: Deterministic model settings for reproducible tests.
    """
    logger.info("Integration test for assay pytest plugin with BradleyTerryEvaluator.")
    await _run_query_generation(context, model_settings)


class LengthEvaluator:
    """
    Evaluates test outputs by comparing the length of novel responses to baseline responses.
    As long as the majority of novel responses are longer than their corresponding baseline responses, the test passes.
    """

    async def __call__(self, input: EvaluatorInput) -> Readout:
        """
        Run pairwise comparison of baseline and novel responses.

        Args:
            input: The evaluator input with baseline dataset and agent responses.

        Returns:
            Readout with passed status and details.
        """
        logger.info("Running length evaluation on captured agent responses")

        if input.baseline_dataset is None:
            raise AssertionError("Baseline dataset is required for LengthEvaluator.")

        # 1. Calculate lengths of baseline and novel responses
        lengths_baseline: list[int] = [len(str(case.expected_output)) for case in input.baseline_dataset.cases]
        lengths_novel: list[int] = []
        for idx, response in enumerate(input.agent_responses):
            if response.output is None:
                logger.warning(f"Response #{idx} has None output.")
                lengths_novel.append(0)
                continue
            lengths_novel.append(len(str(response.output)))

        # 2. Check for length mismatch
        if len(lengths_baseline) != len(lengths_novel):
            error_msg = f"Mismatch in response counts: {len(lengths_baseline)} baseline vs {len(lengths_novel)} novel"
            logger.error(error_msg)
            raise AssertionError(error_msg)

        # 3. Determine wins for novel responses based on length comparison
        wins_novel = [novel > baseline for baseline, novel in zip(lengths_baseline, lengths_novel, strict=True)]
        count_wins = sum(wins_novel)
        count_losses = len(wins_novel) - count_wins

        return Readout(
            passed=count_wins > count_losses,
            details={
                "test_cases_count": len(lengths_baseline),
                "lengths_baseline": lengths_baseline,
                "lengths_novel": lengths_novel,
            },
        )


@pytest.mark.ollama
@pytest.mark.assay(
    generator=generate_evaluation_cases,
    evaluator=LengthEvaluator(),
)
@pytest.mark.asyncio
async def test_integration_lengthevaluator(context: AssayContext, model_settings: ModelSettings) -> None:
    """
    Integration test for a custom evaluator.
    Here we simply demonstrate how to run an assay with a user-defined evaluator which is not shipped with the pytest-assay package.

    An agent generates search queries for various research topics.
    LengthEvaluator then checks whether generated queries are longer than baseline queries.

    Args:
        context: The assay context containing the evaluation dataset context.dataset and other information.
        model_settings: Deterministic model settings for reproducible tests.
    """
    logger.info("Integration test for assay pytest plugin with LengthEvaluator.")
    await _run_query_generation(context, model_settings)
