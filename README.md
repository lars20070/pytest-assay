<div align="center">
  <a href="https://github.com/lars20070/pytest-assay/">
    <picture>
      <img src="https://raw.githubusercontent.com/lars20070/pytest-assay/master/.github/images/banner_pytest-assay.png" alt="pytest-assay" width="60%">
    </picture>
  </a>
</div>

<br>

[![codecov](https://codecov.io/gh/lars20070/pytest-assay/branch/master/graph/badge.svg)](https://codecov.io/gh/lars20070/pytest-assay)
[![Build](https://github.com/lars20070/pytest-assay/actions/workflows/build.yaml/badge.svg)](https://github.com/lars20070/pytest-assay/actions/workflows/build.yaml)
[![Python Version](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/lars20070/pytest-assay/master/pyproject.toml&query=project.requires-python&label=python)](https://www.python.org/downloads/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/lars20070/pytest-assay)
[![PyPI](https://img.shields.io/pypi/v/pytest-assay)](https://pypi.org/project/pytest-assay/)
[![License](https://img.shields.io/github/license/lars20070/pytest-assay)](https://github.com/lars20070/pytest-assay/blob/master/LICENSE)

*pytest-assay* is a framework for the evaluation of [Pydantic AI](https://ai.pydantic.dev) agents. By adding the `@pytest.mark.assay` decorator to a test, you can run an *assay* resulting in a *readout* report which contains the evaluation. The assay compares the current agent responses against pre-recorded baseline responses, e.g. from the `main` branch. The implementation is using pytest hooks which capture [`Agent.run()`](https://ai.pydantic.dev/agent) responses inside the test. Below is a minimal example which evaluates the creativity of a search query generation. For a fully functional example see [`tests/test_plugin_integration.py`](https://github.com/lars20070/pytest-assay/blob/master/tests/test_plugin_integration.py).

```python
def generate_evaluation_cases():
    ...
    return Dataset(cases)


@pytest.mark.assay(
    generator=generate_evaluation_cases,
    evaluator=PairwiseEvaluator(
        model="openai:gpt-4o",
        criterion="Which of the two search queries shows more genuine curiosity and creativity?",
    ),
)
@pytest.mark.asyncio
async def test_query_generation(context: AssayContext):

    agent = Agent(
        model="openai:gpt-4o",
        system_prompt="Generate a concise web search query for the given research topic.",
    )

    for case in context.dataset.cases:
        async with agent:
            result = await agent.run(user_prompt=f"Generate a search query for the following topic: <TOPIC>{case.inputs['topic']}</TOPIC>")
```
Both baseline responses and the final *readout* report are stored in the `pytest-assay/` subfolder.
```
tests
├── pytest-assay
│   └── test_evaluation
│       ├── test_query_generation.json            # Baseline data
│       └── test_query_generation.readout.json    # Readout report
└── test_evaluation.py
```
The baseline data are generated in `new_baseline` mode, and the readout report is generated in default `evaluate` mode.
```bash
uv run pytest tests/test_evaluation.py --assay-mode=new_baseline  # Generates baseline data test_query_generation.json
uv run pytest tests/test_evaluation.py --assay-mode=evaluate      # Generates readout report test_query_generation.readout.json
```