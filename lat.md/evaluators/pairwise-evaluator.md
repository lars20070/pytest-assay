---
lat:
  require-code-mention: true
---
# PairwiseEvaluator

A simpler evaluator that runs one direct A-vs-B comparison per response pair (baseline[i] vs novel[i]). Passes if novel wins more comparisons than it loses.

## How it works

Pairs each baseline response with its novel counterpart and runs one direct A-vs-B comparison per pair.

1. Baseline `expected_output` strings are paired 1-to-1 with novel `AgentRunResult.output` strings.
2. For each pair, a judge LLM receives a prompt with `<A>` (baseline) and `<B>` (novel) and picks the better one.
3. `passed = wins_novel > losses_novel` (strict majority).
4. Raises `AssertionError` if baseline and novel counts differ.

## Configuration

Constructor parameters for `PairwiseEvaluator`.

```python
PairwiseEvaluator(
    model=None,                             # defaults to Ollama qwen3:8b
    criterion="Which of the two responses is better?",
)
```

## Readout details

Example JSON serialized to `<assay_path>.readout.json` after a pairwise evaluation.

```json
{
  "test_cases_count": 5,
  "wins_baseline": [false, true, false, false, false],
  "wins_novel":    [true, false, true, true, true]
}
```

## Comparison with BradleyTerry

Side-by-side comparison of the two built-in evaluators across key operational dimensions.

| | PairwiseEvaluator | BradleyTerryEvaluator |
|---|---|---|
| Game count | exactly `n` | `O(n log n)` to `n(n-1)` |
| Handles position bias | no | yes (plays both `(i,j)` and `(j,i)`) |
| Requires equal counts | yes | no |
| Score type | win/loss counts | continuous Bradley-Terry strength |
| Default? | no | yes |