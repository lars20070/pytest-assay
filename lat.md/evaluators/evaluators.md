---
lat:
  require-code-mention: true
---
# Evaluators

Evaluators compare captured agent responses against a stored baseline and produce a `Readout`. They are async callables conforming to the `Evaluator` protocol defined in [[evaluators#Evaluator Protocol]].

The plugin calls the evaluator in `_run_evaluation` via `asyncio.run(evaluator(eval_input))`.

## Evaluator Protocol

```python
class Evaluator(Protocol):
    def __call__(self, input: EvaluatorInput) -> Coroutine[Any, Any, Readout]: ...
```

Any async callable accepting `EvaluatorInput` and returning `Readout` satisfies the protocol. Configuration is done at instantiation time; the plugin only interacts with `__call__`.

## EvaluatorInput

Passed by the plugin to every evaluator:

| Field | Type | Description |
|---|---|---|
| `baseline_dataset` | <code>Dataset &#124; None</code> | Deep-copied snapshot from the loaded assay JSON. `None` when no baseline file exists. |
| `agent_responses` | `list[AgentRunResult]` | Responses captured by monkeypatching `Agent.run()` during the test run. |

## Readout

Returned by every evaluator and serialized to `<assay_path>.readout.json`:

| Field | Type | Description |
|---|---|---|
| `passed` | `bool` | Whether novel responses outperformed the baseline. Defaults to `True`. |
| `details` | <code>dict &#124; None</code> | Evaluator-specific structured data (scores, win counts, etc.). |

## Built-in Evaluators

The framework ships with two built-in evaluators:

- [[bradleyterry---evaluator]] — A tournament-based evaluator producing continuous strength scores.
- [[pairwise-evaluator]] — A simple A-vs-B evaluator producing binary win/loss results.
