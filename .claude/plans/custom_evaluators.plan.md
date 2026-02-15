# Custom Evaluators Design Plan

## Problem

A user writing `MyCustomEvaluator` would need to do:

```python
from pytest_assay.plugin import AGENT_RESPONSES_KEY, BASELINE_DATASET_KEY
```

These stash keys are **internal implementation details** -- not exported in `__all__`, and they couple any custom evaluator directly to the plugin's storage mechanism. The `Evaluator` protocol receiving a raw `pytest.Item` is the root cause: it forces evaluators to know *how* to extract data rather than just *receiving* it.

## Design: Pass an `EvaluatorInput` Instead of `Item`

Introduce a data class that the plugin constructs and passes to evaluators, replacing the raw `Item`:

```python
# models.py
class EvaluatorInput(BaseModel):
    baseline_dataset: Dataset | None
    agent_responses: list[AgentRunResult[Any]]

class Evaluator(Protocol):
    def __call__(self, input: EvaluatorInput) -> Coroutine[Any, Any, Readout]: ...
```

The plugin's `_run_evaluation` would change from:

```python
readout = asyncio.run(evaluator(item))
```

to:

```python
eval_input = EvaluatorInput(
    baseline_dataset=item.stash.get(BASELINE_DATASET_KEY, None),
    agent_responses=item.stash.get(AGENT_RESPONSES_KEY, []),
)
readout = asyncio.run(evaluator(eval_input))
```

The stash keys stay private. Evaluators receive exactly the data they need.

## What Changes

| Component | Change |
|---|---|
| `Evaluator` protocol | `__call__(self, input: EvaluatorInput)` instead of `__call__(self, item: Item)` |
| `_run_evaluation` in plugin.py | Build `EvaluatorInput` from stash, pass it |
| `PairwiseEvaluator.__call__` | Accept `EvaluatorInput`, drop the stash key import |
| `BradleyTerryEvaluator.__call__` | Accept `EvaluatorInput`, drop the stash key import |
| Public exports | Add `EvaluatorInput`, `Evaluator`, `Readout` to `__all__` |

## What a Custom Evaluator Looks Like

```python
from pytest_assay.models import Evaluator, EvaluatorInput, Readout

class MyCustomEvaluator:
    async def __call__(self, input: EvaluatorInput) -> Readout:
        # input.baseline_dataset has the baseline cases
        # input.agent_responses has the captured AgentRunResult objects
        score = my_scoring_logic(input.baseline_dataset, input.agent_responses)
        return Readout(passed=score > 0.5, details={"score": score})
```

No internal imports needed. Fully decoupled from pytest stash mechanics.

## Why Not Alternatives

- **Export the stash keys as public API**: Leaks an implementation detail. If you ever change how data flows (e.g. stop using stash), every custom evaluator breaks.
- **Accessor helper functions** (`get_baseline(item)`, `get_responses(item)`): Better than raw keys, but still couples evaluators to `pytest.Item`. Harder to unit-test evaluators in isolation.
- **The `EvaluatorInput` approach**: Evaluators become plain async functions on data. Easy to test, no pytest dependency, no internal imports. The plugin owns the extraction logic.

## Open Questions

1. Should `EvaluatorInput` also carry metadata (test name, assay path, mode) so evaluators can customize behavior per-test?
2. `AgentRunResult` comes from `pydantic-ai` -- is that an acceptable public API surface, or should responses be further abstracted (e.g. to plain strings)?
