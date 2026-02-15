#!/usr/bin/env python3
from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Literal

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from pytest_assay.config import config
from pytest_assay.logger import logger

if TYPE_CHECKING:
    from pytest import Item

    from pytest_assay.models import Readout


EVALUATION_INSTRUCTIONS = """
You are presented with a question and two possible answers A and B. Evaluate carefully whether answer A or answer B is the better reply.
You have got only these two options. Each comparison should be independent but internally consistent.

<EXAMPLES>
Example 1:
<QUESTION> Which of the two ice cream flavours below is more creative? </QUESTION>
<A> Vanilla </A>
<B> Pickled Citrus Ribbon </B>
Expected output:
{
    "response": "B"
}

Example 2:
<QUESTION> Which search query shows more genuine curiosity? </QUESTION>
<A> effect of ocean acidification feedback loops on Arctic methane release </A>
<B> climate change effects </B>
Expected output:
{
    "response": "A"
}

Example 3:
<QUESTION> Which reply is more insulting? </QUESTION>
<A> Your argument lacks logical coherence and fails to address the core issue at hand. </A>
<B> That's an interesting perspective, though I see it differently. </B>
Expected output:
{
    "response": "A"
}
</EXAMPLES>

<REQUIREMENTS>
1. Consider the question carefully. What aspects are important for the answer?
2. Think about answer A. Is it a good answer to the question? Why (not)?
3. Think about answer B. Is it a good answer to the question? Why (not)?
4. Make a decision based on your analysis.
</REQUIREMENTS>

<OUTPUT_FORMAT>
You must respond with valid JSON containing exactly one field called "response" with value "A" or "B":

{
    "response": "A"
}

or

{
    "response": "B"
}

Do NOT include explanations, reasoning, or any other fields.
</OUTPUT_FORMAT>
"""


class PairwiseEvaluator:
    """
    Evaluates test outputs using pairwise comparison.

    Configuration is set at instantiation; __call__ runs the evaluation.
    """

    def __init__(
        self,
        model: str | OpenAIChatModel | None = None,
        criterion: str = "Which of the two responses is better?",
    ) -> None:
        """
        Configure the evaluator.

        Args:
            model: The language model or model string to use for evaluation. Defaults to qwen3:8b on Ollama.
            criterion: The evaluation criterion for pairwise comparison.
        """
        if model is None:
            self.model = OpenAIChatModel(
                model_name=config.ollama_model,
                provider=OpenAIProvider(base_url=f"{config.ollama_base_url}/v1"),
            )
        else:
            self.model = model
        self.model_settings = ModelSettings(
            temperature=0.0,
            timeout=300,
        )
        self.system_prompt = EVALUATION_INSTRUCTIONS
        self.agent = Agent(
            model=self.model,
            output_type=Literal["A", "B"],
            system_prompt=self.system_prompt,
            retries=5,
            instrument=True,
        )
        self.criterion = criterion

    async def __call__(self, item: Item) -> Readout:
        """
        Run pairwise comparison of baseline and novel responses.

        Args:
            item: The pytest test item with assay context and captured responses.

        Returns:
            Readout with passed status and details.
        """
        from pytest_assay.models import Readout  # noqa: PLC0415
        from pytest_assay.plugin import AGENT_RESPONSES_KEY, BASELINE_DATASET_KEY  # noqa: PLC0415

        logger.info("Running Pairwise evaluation on captured agent responses")

        # 1. Baseline responses from previously serialized assay dataset
        responses_baseline: list[str] = []
        baseline_dataset = item.stash.get(BASELINE_DATASET_KEY, None)
        if baseline_dataset is not None:
            for idx, case in enumerate(baseline_dataset.cases):
                logger.debug(f"Baseline response #{idx}: {repr(case.expected_output)[:100]}")
                responses_baseline.append(str(case.expected_output))

        # 2. Novel responses from current test run
        responses_novel: list[str] = []
        responses = item.stash.get(AGENT_RESPONSES_KEY, [])
        for idx, response in enumerate(responses):
            if response.output is None:
                logger.warning(f"Response #{idx} has None output.")
                continue
            logger.debug(f"Novel response #{idx}: {repr(response.output)[:100]}")
            responses_novel.append(response.output)

        if len(responses_baseline) != len(responses_novel):
            error_msg = f"Mismatch in response counts: {len(responses_baseline)} baseline vs {len(responses_novel)} novel"
            logger.error(error_msg)
            raise AssertionError(error_msg)

        # 3. Loop over all response pairs and compare them
        wins_novel = list[bool]()
        for idx, (baseline, novel) in enumerate(zip(responses_baseline, responses_novel, strict=True)):
            logger.info(f"Comparing response pair #{idx}")

            prompt = textwrap.dedent(f"""
                <QUESTION> {self.criterion} </QUESTION>
                <A> {baseline} </A>
                <B> {novel} </B>
            """)
            logger.debug(f"Pairwise comparison prompt for pair #{idx}:\n{prompt}")

            async with self.agent:
                result = await self.agent.run(
                    user_prompt=prompt,
                    model_settings=self.model_settings,
                )
            logger.debug(f"Pairwise comparison result for pair #{idx}: {result.output}")

            if result.output == "A":
                wins_novel.append(False)
            else:
                wins_novel.append(True)

        # 4. Generate readout
        wins_count = sum(wins_novel)
        losses_count = len(wins_novel) - wins_count
        return Readout(
            passed=wins_count > losses_count,
            details={
                "test_cases_count": len(responses_baseline),
                "wins_baseline": [not win for win in wins_novel],
                "wins_novel": wins_novel,
            },
        )
