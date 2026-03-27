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

## BradleyTerryEvaluator

The default evaluator. Runs a pairwise tournament across all baseline and novel responses, computes Bradley-Terry strength scores, and passes if the average novel score exceeds the average baseline score.

### How it works

Converts all responses to players, orchestrates a tournament, and derives Bradley-Terry strength scores.

1. All baseline `expected_output` strings become `EvalPlayer` instances (indices `0..n-1`).
2. All novel `AgentRunResult.output` strings become further `EvalPlayer` instances (indices `n..2n-1`).
3. An `EvalGame` pits pairs of players against each other: a judge LLM picks `"A"` or `"B"` given the configured `criterion`.
4. An `EvalTournament` orchestrates games using a `TournamentStrategy`. Defaults to [[evaluators#Tournament Strategies#Adaptive Uncertainty Strategy]].
5. Bradley-Terry scores are estimated from the game outcomes using `choix.ep_pairwise`.
6. `passed = avg_novel_score > avg_baseline_score`.

### Configuration

```python
BradleyTerryEvaluator(
    model=None,                   # defaults to Ollama qwen3:8b via OpenAI-compatible API
    criterion="Which of the two agent responses is better?",
    max_standard_deviation=2.0,   # convergence threshold for adaptive strategy
)
```

The judge agent uses `temperature=0.0` and `retries=5` for deterministic, robust comparisons. `output_type=Literal["A", "B"]` enforces structured output. The default tournament strategy is [[evaluators#Tournament Strategies#Adaptive Uncertainty Strategy]].

### Readout details

Example JSON serialized to `<assay_path>.readout.json` after a BradleyTerry evaluation.

```json
{
  "test_cases_count": 10,
  "scores_baseline": [0.12, -0.34, ...],
  "scores_novel": [0.45, 0.11, ...]
}
```

## PairwiseEvaluator

A simpler evaluator that runs one direct A-vs-B comparison per response pair (baseline[i] vs novel[i]). Passes if novel wins more comparisons than it loses.

### How it works

Pairs each baseline response with its novel counterpart and runs one direct A-vs-B comparison per pair.

1. Baseline `expected_output` strings are paired 1-to-1 with novel `AgentRunResult.output` strings.
2. For each pair, a judge LLM receives a prompt with `<A>` (baseline) and `<B>` (novel) and picks the better one.
3. `passed = wins_novel > losses_novel` (strict majority).
4. Raises `AssertionError` if baseline and novel counts differ.

### Configuration

Constructor parameters for `PairwiseEvaluator`.

```python
PairwiseEvaluator(
    model=None,                             # defaults to Ollama qwen3:8b
    criterion="Which of the two responses is better?",
)
```

### Readout details

Example JSON serialized to `<assay_path>.readout.json` after a pairwise evaluation.

```json
{
  "test_cases_count": 5,
  "wins_baseline": [false, true, false, false, false],
  "wins_novel":    [true, false, true, true, true]
}
```

### Comparison with BradleyTerry

Side-by-side comparison of the two built-in evaluators across key operational dimensions.

| | PairwiseEvaluator | BradleyTerryEvaluator |
|---|---|---|
| Game count | exactly `n` | `O(n log n)` to `n(n-1)` |
| Handles position bias | no | yes (plays both `(i,j)` and `(j,i)`) |
| Requires equal counts | yes | no |
| Score type | win/loss counts | continuous Bradley-Terry strength |
| Default? | no | yes |

## Tournament Strategies

Used exclusively by `BradleyTerryEvaluator`. All strategies share the signature:

```python
async def strategy(
    players: list[EvalPlayer],
    game: EvalGame,
    agent: Agent,
    model_settings: ModelSettings,
    **kwargs,
) -> list[EvalPlayer]: ...
```

### Adaptive Uncertainty Strategy

The default strategy. Two phases:

**Bootstrap phase** — plays `max(2n, n/2 · log(n))` random games to ensure the comparison graph is strongly connected (Erdős-Rényi 1960 threshold).

**Optimization phase** — iteratively selects the player pair with the highest score uncertainty:

```
Var(s_i - s_j) = Var(s_i) + Var(s_j) - 2·Cov(s_i, s_j)
```

Stops when `sqrt(max_uncertainty) < max_standard_deviation` or all pairs have been played. Uses `choix.ep_pairwise` for score and covariance estimation.

Parameters: `max_standard_deviation` (default `2.0`), `alpha` prior strength (default `0.1`).

### Random Sampling Strategy

Plays all `n(n-1)` directed games in random order (or a `fraction_of_games` subset). Simple exhaustive baseline; use when you want maximum coverage regardless of game count. Scores via `choix.ilsr_pairwise`.

### Round Robin Strategy

Each player plays `number_of_rounds` games against randomly chosen opponents. Scores via `choix.ilsr_pairwise`. Lower game count than random sampling but less coverage.

## Internal Models

Pydantic models used internally by the BradleyTerry evaluator pipeline.

### EvalPlayer

`idx: int` — unique tournament index. `item: str` — the response text. `score: float | None` — Bradley-Terry strength, populated after tournament.

### EvalGame

Holds the `criterion` string and runs a single pairwise comparison via `Agent.run()`. Returns `(winner_idx, loser_idx)`.

### EvalTournament

Holds `players` and `game`, delegates to a `TournamentStrategy` callable. Exposes `get_player_by_idx()` for post-tournament score retrieval.
