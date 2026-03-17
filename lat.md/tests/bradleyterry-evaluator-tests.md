---
lat:
  require-code-mention: true
---
# BradleyTerry Evaluator Tests

Unit and integration tests in `tests/evaluators/test_bradleyterry.py`.

## EvalPlayer

- Created with required fields; `score` defaults to `None`.
- Created with explicit `score`.
- `score` is mutable after creation.
- Negative `idx` is permitted (it is an identifier, not a position).
- Empty `item` string is permitted.

## EvalGame

- `criterion` is stored on the model.
- Returns `(player_A_idx, player_B_idx)` when the agent picks `"A"`.
- Returns `(player_B_idx, player_A_idx)` when the agent picks `"B"`.
- The prompt sent to the agent contains both player items and the criterion string.
- Result tuple uses actual `idx` values, not positional indices.

## EvalTournament

- Constructed with the correct number of players and the game criterion.
- `get_player_by_idx` returns the player with the matching `idx` and correct `item`.
- `get_player_by_idx` raises `ValueError` for an unknown `idx`.
- When no strategy is provided, `adaptive_uncertainty_strategy` is invoked.
- A custom strategy function receives `(players, game, agent, model_settings)` positional args.
- Extra kwargs (e.g., `max_standard_deviation`, `alpha`) are forwarded to the strategy.
- After `run()`, `tournament.players` is updated to the scored players returned by the strategy.

## BradleyTerryEvaluator

- Default init: `OpenAIChatModel` on Ollama, `temperature=0.0`, `timeout=300`, default `criterion` and `max_standard_deviation=2.0`.
- Custom `criterion` and `max_standard_deviation` stored.
- Custom `OpenAIChatModel` instance is used and reflected on the internal agent.
- Model string (e.g., `"openai:gpt-4o-mini"`) accepted.
- Empty player list returns `Readout(passed=True, details={"message": "No players to evaluate"})`.
- With one baseline case and one novel response, creates a tournament with the correct criterion and `max_standard_deviation`, returns a `Readout`.
- Conforms to `Evaluator` protocol: callable, `__call__` is a coroutine function.

## BradleyTerry Strategy Integration (ollama)

Requires Ollama. All integration tests are marked `@pytest.mark.ollama`.

- `EvalGame` integration: running a game between vanilla and a creative ice cream flavour returns the expected winner index.
- `EvalTournament` integration: running the default strategy scores all players with non-`None` float scores; running `random_sampling_strategy` with `fraction_of_games=0.3` also scores all players.
- `random_sampling_strategy` parametrized over `fraction_of_games=None`, `0.3`, and an out-of-range value; all produce float scores for every player.
- `round_robin_strategy` with `number_of_rounds=1` produces float scores for every player.
- `adaptive_uncertainty_strategy` with `max_standard_deviation=1.0` and `alpha=0.01` produces float scores for every player.
