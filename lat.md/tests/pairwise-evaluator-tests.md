---
lat:
  require-code-mention: true
---
# Pairwise Evaluator Tests

Unit tests in `tests/evaluators/test_pairwise.py`.

## PairwiseEvaluator

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
