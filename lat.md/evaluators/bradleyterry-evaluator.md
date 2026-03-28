---
lat:
  require-code-mention: true
---
# BradleyTerryEvaluator

The default evaluator. Runs a tournament of pairwise games across all baseline and novel responses, computes [Bradley-Terry strength scores](https://en.wikipedia.org/wiki/Bradley–Terry_model), and passes if the average novel score exceeds the average baseline score.

## How it works

Converts all responses to players, orchestrates a tournament, and derives Bradley-Terry strength scores.

1. All baseline `expected_output` strings become `EvalPlayer` instances (indices `0..n-1`).
2. All novel `AgentRunResult.output` strings become further `EvalPlayer` instances (indices `n..2n-1`).
3. An `EvalGame` pits pairs of players against each other: a judge LLM picks `"A"` or `"B"` given the configured `criterion`.
4. An `EvalTournament` orchestrates games using a `TournamentStrategy`. Defaults to [[bradleyterry-evaluator#Tournament Strategies#Adaptive Uncertainty Strategy]].
5. Bradley-Terry scores are estimated from the game outcomes using `choix.ep_pairwise`.
6. `passed = avg_novel_score > avg_baseline_score`.

## Configuration

```python
BradleyTerryEvaluator(
    model=None,                   # defaults to Ollama qwen3:8b via OpenAI-compatible API
    criterion="Which of the two agent responses is better?",
    max_standard_deviation=2.0,   # convergence threshold for adaptive strategy
)
```

The judge agent uses `temperature=0.0` and `retries=5` for deterministic, robust comparisons. `output_type=Literal["A", "B"]` enforces structured output. The default tournament strategy is [[bradleyterry-evaluator#Tournament Strategies#Adaptive Uncertainty Strategy]].

## Readout details

Example JSON serialized to `<assay_path>.readout.json` after a Bradley-Terry evaluation.

```json
{
  "test_cases_count": 10,
  "scores_baseline": [0.12, -0.34, ...],
  "scores_novel": [0.45, 0.11, ...]
}
```

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

Pydantic models used internally by the Bradley-Terry evaluator pipeline.

### EvalPlayer

* `idx: int` — unique tournament index.
* `item: str` — the response text.
* `score: float | None` — Bradley-Terry strength, populated after tournament.

### EvalGame

Holds the `criterion` string and runs a single pairwise comparison via `Agent.run()`. Returns `(winner_idx, loser_idx)`.

### EvalTournament

Holds `players` and `game`, delegates to a `TournamentStrategy` callable. Exposes `get_player_by_idx()` for post-tournament score retrieval.