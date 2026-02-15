#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

import choix
import numpy as np
from pydantic import BaseModel, Field
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
You have got only these two options. Your evaluations contribute to Bradley-Terry scores across multiple items. Consistency and
objectivity are critical for reliable rankings. Each comparison should be independent but internally consistent.

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


EVALUATION_GAME_PROMPT = "<QUESTION> {criterion} </QUESTION>\n<A> {a} </A>\n<B> {b} </B>"


class EvalPlayer(BaseModel):
    idx: int = Field(..., description="unique identifier for the player")
    item: str = Field(..., description="item to be scored")
    score: float | None = Field(default=None, description="Bradley-Terry strength score for the item")


class EvalGame(BaseModel):
    criterion: str = Field(..., description="evaluation criterion on which players should be judged")

    async def run(self, players: tuple[EvalPlayer, EvalPlayer], agent: Agent[None, Any], model_settings: ModelSettings) -> tuple[int, int]:
        prompt = EVALUATION_GAME_PROMPT.format(criterion=self.criterion, a=players[0].item, b=players[1].item)

        async with agent:
            result = await agent.run(
                user_prompt=prompt,
                model_settings=model_settings,
            )

        if result.output == "A":
            return (players[0].idx, players[1].idx)
        else:
            return (players[1].idx, players[0].idx)


TournamentStrategy = Callable[
    [list[EvalPlayer], EvalGame, Agent[None, Any], ModelSettings],
    Awaitable[list[EvalPlayer]],
]


async def random_sampling_strategy(
    players: list[EvalPlayer],
    game: EvalGame,
    agent: Agent[None, Any],
    model_settings: ModelSettings,
    fraction_of_games: float | None = None,
) -> list[EvalPlayer]:
    """
    Random sampling tournament strategy.

    In a tournament with n players, there are n*(n-1) possible pairwise games. We consider
    (i, j) and (j, i) as different games in order to ensure that the evaluation agent does
    not introduce any ordering bias. The strategy plays all possible games in random order.
    The strategy is simple and not efficient. But when all games are played, it returns the
    best possible scores.

    Args:
        players: List of players in the tournament.
        game: Game defining the pairwise comparisons.
        agent: Agent for the game.
        model_settings: Model settings for the game.
        fraction_of_games: Fraction of all possible games to be played. Between 0 and 1. If None, all games are played.

    Returns:
        List of players with Bradley-Terry scores.
    """
    scoreboard: list[tuple[int, int]] = []

    # Generate all possible games
    n = len(players)
    matches = [(i, j) for i in range(n) for j in range(n) if i != j]
    random.shuffle(matches)
    if fraction_of_games is not None and 0 < fraction_of_games <= 1:
        number_of_games = int(len(matches) * fraction_of_games)
        matches = matches[:number_of_games]
    logger.info(f"Random sampling strategy: {n} players playing {len(matches)} games")

    # Play all games
    for i, match in enumerate(matches):
        player_1, player_2 = players[match[0]], players[match[1]]
        logger.debug(f"Game {i + 1} / {len(matches)}: Player {player_1.idx} vs Player {player_2.idx}")

        result = await game.run(
            players=(player_1, player_2),
            agent=agent,
            model_settings=model_settings,
        )
        scoreboard.append(result)
        logger.debug(f"Result: {result}")

    # Calculate Bradley-Terry scores and update players
    scores = choix.ilsr_pairwise(len(players), scoreboard, alpha=0.01)
    for i, player in enumerate(players):
        player.score = float(scores[i])

    return players


async def round_robin_strategy(
    players: list[EvalPlayer],
    game: EvalGame,
    agent: Agent[None, Any],
    model_settings: ModelSettings,
    number_of_rounds: int = 2,
) -> list[EvalPlayer]:
    """
    Round-robin tournament strategy.

    Each player plays against a randomly selected opponent for a given number of rounds.
    The scores are calculated from the game outcomes using the Bradley-Terry algorithm.
    The strategy ensures that each player plays at least number_of_rounds games.
    The strategy is simple but not efficient.

    Args:
        players: List of players in the tournament.
        game: Game defining the pairwise comparisons.
        agent: Agent for the game.
        model_settings: Model settings for the game.
        number_of_rounds: Number of rounds.

    Returns:
        List of players with Bradley-Terry scores.
    """
    scoreboard: list[tuple[int, int]] = []

    logger.info(f"Round-robin strategy: {len(players)} players, {number_of_rounds} rounds")

    for n in range(number_of_rounds):
        logger.debug(f"Starting round {n + 1} / {number_of_rounds}")

        for player in players:
            # Pick a random opponent (excluding self)
            idx = random.randrange(len(players))
            while idx == player.idx:
                idx = random.randrange(len(players))
            player_2 = players[idx]
            logger.debug(f"Game: Player {player.idx} vs Player {player_2.idx}")

            # Play the game
            result = await game.run(
                players=(player, player_2),
                agent=agent,
                model_settings=model_settings,
            )
            scoreboard.append(result)
            logger.debug(f"Result: {result}")

    # Calculate Bradley-Terry scores and update players
    scores = choix.ilsr_pairwise(len(players), scoreboard, alpha=0.01)
    for i, player in enumerate(players):
        player.score = float(scores[i])

    return players


async def adaptive_uncertainty_strategy(
    players: list[EvalPlayer],
    game: EvalGame,
    agent: Agent[None, Any],
    model_settings: ModelSettings,
    max_standard_deviation: float = 2.0,
    alpha: float = 0.1,
) -> list[EvalPlayer]:
    """
    Adaptive uncertainty tournament strategy.

    The strategy consists of two phases:
    (1) Bootstrap phase: The Bradley-Terry model requires the comparison graph to be strongly connected i.e.
        there must be a path between any two players. We therefore start by playing n/2*log(n) random games where
        n is the number of players. See Erdős-Rényi 1960. With fewer games, any scores are likely to be unreliable.
    (2) Optimization phase: In this phase, we iteratively calculate the Bradley-Terry scores and their
        covariance matrix, and play the game for which the player scores are the most uncertain.

        Let s_i and s_j the Bradley-Terry scores of players i and j respectively. The uncertainty in their
        relative strength is then given by

        Var(s_i - s_j) = Var(s_i) + Var(s_j) - 2*Cov(s_i, s_j)

        We stop when the standard deviation sqrt(Var(s_i - s_j)) of the most uncertain pair drops below
        the threshold max_standard_deviation, or when all possible pairs have been played.

    Comment on max_standard_deviation parameter:
        Typically, a standard deviation below 1.0 is a good stopping condition. However, the uncertainty
        depends greatly on the evaluation problem. For a problem such as "Which of the following ice cream
        flavours is the most creative one? Vanilla or Chocolate or Strawberry?", the uncertainty will remain
        high even after many games.

    Comment on alpha parameter:
        The alpha parameter is the prior strength for the Bradley-Terry model. Higher alpha (e.g. 0.8) is a
        strong prior towards equal player strengths. The games have a smaller influence on the scores, and
        the scores remain close to the mean of 0. Lower alpha (e.g. 0.1) on the other hand lets the games
        influence the scores more strongly. However, for a sparse comparison graph, the scores can become
        less stable. Typical values are between 0.1 and 0.3.

    Args:
        players: List of players in the tournament.
        game: Game defining the pairwise comparisons.
        agent: Agent for the game.
        model_settings: Model settings for the game.
        max_standard_deviation: Maximum standard deviation for the most uncertain pair. See also above.
        alpha: Prior strength for the Bradley-Terry model. Between 0 and 1. See also above.

    Returns:
        List of players with Bradley-Terry scores.
    """
    scoreboard: list[tuple[int, int]] = []
    n = len(players)

    logger.info(f"Adaptive uncertainty strategy: {n} players")

    # (1) Bootstrap phase
    number_of_bootstrap_games = max(2 * n, int(n / 2 * np.log(n)))
    matches = [(i, j) for i in range(n) for j in range(n) if i != j]
    random.shuffle(matches)
    matches = matches[:number_of_bootstrap_games]
    for idx, match in enumerate(matches):
        player_1, player_2 = players[match[0]], players[match[1]]
        logger.debug(f"Bootstrap game {idx + 1} / {len(matches)}: Player {player_1.idx} vs Player {player_2.idx}")

        result = await game.run(
            players=(player_1, player_2),
            agent=agent,
            model_settings=model_settings,
        )
        scoreboard.append(result)
        logger.debug(f"Result: {result}")

    # (2) Optimization phase
    max_number_of_games = int(n * (n - 1) / 2.0)
    previous_scores: np.ndarray | None = None
    for idx in range(max_number_of_games):
        logger.debug(f"Optimization game {idx + 1} / {max_number_of_games}")

        # Calculate the Bradley-Terry scores and covariance matrix
        scores, cov_matrix = choix.ep_pairwise(n_items=n, data=scoreboard, alpha=alpha, model="logit")

        # For monitoring only, check the absolute score changes.
        # Note that this change is not decreasing monotonically.
        if previous_scores is not None:
            absolute_change = np.abs(scores - previous_scores)
            max_change = np.max(absolute_change)
            logger.debug(f"Maximum absolute change in the scores since last iteration: {max_change:.4f}")
        previous_scores = scores.copy()

        # Find most uncertain pair which has not yet been played.
        max_uncertainty = -1.0
        next_pair: tuple[int, int] | None = None
        for i in range(n):
            for j in range(i + 1, n):
                # Check if the pair has already been played.
                # Here we assume that games are symmetric which is not quite correct but good enough.
                if (players[i].idx, players[j].idx) in scoreboard or (players[j].idx, players[i].idx) in scoreboard:
                    continue

                # Uncertainty of the pair
                uncertainty = cov_matrix[i, i] + cov_matrix[j, j] - 2 * cov_matrix[i, j]
                if uncertainty > max_uncertainty:
                    max_uncertainty = uncertainty
                    next_pair = (i, j)

        # Terminate optimization phase?
        if next_pair is None:
            logger.info("All pairs have been played. Ending optimization phase.")
            break
        if math.sqrt(max_uncertainty) < max_standard_deviation:
            logger.info(
                f"The standard deviation of the most uncertain pair {math.sqrt(max_uncertainty)} is below the threshold {max_standard_deviation}. "
                f"Ending optimization phase."
            )
            break

        # Play the most uncertain pair
        logger.debug(
            f"Most uncertain pair: Player {players[next_pair[0]].idx} vs Player {players[next_pair[1]].idx} "
            f"(std dev: {math.sqrt(max_uncertainty):.4f}, std dev termination target: {max_standard_deviation:.4f})"
        )
        player_1, player_2 = players[next_pair[0]], players[next_pair[1]]
        result = await game.run(
            players=(player_1, player_2),
            agent=agent,
            model_settings=model_settings,
        )
        scoreboard.append(result)
        logger.debug(f"Result: {result}")

    # Final calculation of Bradley-Terry scores and update players
    scores, _ = choix.ep_pairwise(n_items=n, data=scoreboard, alpha=alpha, model="logit")
    for i, player in enumerate(players):
        player.score = float(scores[i])

    return players


class EvalTournament(BaseModel):
    game: EvalGame = Field(..., description="game to be played in the tournament")
    players: list[EvalPlayer] = Field(..., description="players participating in the tournament")

    def get_player_by_idx(self, idx: int) -> EvalPlayer:
        """
        Return player with unique identifier idx

        Args:
            idx: Unique identifier of the player.

        Returns:
            Player with the specified unique identifier.
        """
        for player in self.players:
            if player.idx == idx:
                return player
        raise ValueError(f"Player with unique identifier {idx} not found.")

    async def run(
        self,
        agent: Agent[None, Any],
        model_settings: ModelSettings,
        strategy: TournamentStrategy | None = None,
        **strategy_kwargs: Any,  # noqa: ANN401
    ) -> list[EvalPlayer]:
        """
        Runs the evaluation tournament using the specified strategy.

        The strategy function handles game sampling, game execution and scoring
        allowing complete flexibility in the tournament algorithms.

        Args:
            agent: Agent for the evaluation game.
            model_settings: Model settings for the evaluation game.
            strategy: Function with the tournament algorithm. Defaults to the adaptive uncertainty strategy.
            **strategy_kwargs: Additional arguments passed to the strategy function.

        Returns:
            List of players with scores.
        """
        # Use default strategy if none provided
        if strategy is None:
            strategy = adaptive_uncertainty_strategy

        logger.info(f"Starting tournament with {len(self.players)} players using {strategy.__name__}")

        # Run the tournament strategy (returns players with scores)
        self.players = await strategy(
            self.players,
            self.game,
            agent,
            model_settings,
            **strategy_kwargs,
        )

        logger.info(f"Tournament complete using {strategy.__name__}")
        return self.players


class BradleyTerryEvaluator:
    """
    Evaluates test outputs using Bradley-Terry tournament scoring.

    Configuration is set at instantiation; __call__ runs the evaluation.
    """

    def __init__(
        self,
        model: str | OpenAIChatModel | None = None,
        criterion: str = "Which of the two agent responses is better?",
        max_standard_deviation: float = 2.0,
    ) -> None:
        """
        Configure the evaluator.

        Args:
            model: The language model or model string to use for evaluation.
            criterion: The evaluation criterion for pairwise comparison.
            max_standard_deviation: Convergence threshold for adaptive strategy.
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
        self.max_standard_deviation = max_standard_deviation

    async def __call__(self, item: Item) -> Readout:
        """
        Run Bradley-Terry tournament on baseline and novel responses.

        Args:
            item: The pytest test item with assay context and captured responses.

        Returns:
            Readout with passed status and details.
        """
        from pytest_assay.models import Readout  # noqa: PLC0415
        from pytest_assay.plugin import AGENT_RESPONSES_KEY, BASELINE_DATASET_KEY  # noqa: PLC0415

        logger.info("Running Bradley-Terry evaluation on captured agent responses")

        # Prepare the list of all players, baseline and novel
        players: list[EvalPlayer] = []

        # 1. Baseline players from previously serialized assay dataset
        baseline_dataset = item.stash.get(BASELINE_DATASET_KEY, None)
        baseline_case_count = 0
        if baseline_dataset is not None:
            for idx, case in enumerate(baseline_dataset.cases):
                players.append(EvalPlayer(idx=idx, item=str(case.expected_output)))
            baseline_case_count = len(baseline_dataset.cases)

        # 2. Novel players from current test run
        responses = item.stash.get(AGENT_RESPONSES_KEY, [])
        for idx, response in enumerate(responses):
            if response.output is None:
                logger.warning(f"Response #{idx} has None output.")
                continue
            players.append(EvalPlayer(idx=idx + baseline_case_count, item=response.output))

        # Log all players before tournament
        for player in players:
            logger.debug(f"Player #{player.idx} item: {repr(player.item)[:100]}")

        if not players:
            logger.debug("No players to evaluate in tournament.")
            return Readout(passed=True, details={"message": "No players to evaluate"})

        # Run the Bradley-Terry tournament to score both baseline and novel queries
        game = EvalGame(criterion=self.criterion)
        tournament = EvalTournament(players=players, game=game)
        players_scored = await tournament.run(
            agent=self.agent,
            model_settings=self.model_settings,
            strategy=adaptive_uncertainty_strategy,
            max_standard_deviation=self.max_standard_deviation,
        )

        # Players sorted by score
        players_sorted = sorted(players_scored, key=lambda p: p.score if p.score is not None else float("-inf"))
        for player in players_sorted:
            logger.debug(f"Player {player.idx:4d}   score: {player.score:7.4f}   query: {player.item}")

        # Average score for both baseline and novel queries
        scores_baseline = [tournament.get_player_by_idx(idx=i).score or 0.0 for i in range(len(players) // 2)]
        scores_novel = [tournament.get_player_by_idx(idx=i + len(players) // 2).score or 0.0 for i in range(len(players) // 2)]

        if scores_baseline and scores_novel:
            avg_baseline = np.mean(scores_baseline)
            avg_novel = np.mean(scores_novel)
            passed = bool(avg_novel > avg_baseline)
            logger.debug(f"Average score for baseline queries (Players 0 to 9): {avg_baseline:7.4f}")
            logger.debug(f"Average score for novel queries  (Players 10 to 19): {avg_novel:7.4f}")
        else:
            passed = False

        return Readout(
            passed=passed,
            details={
                "test_cases_count": len(players) // 2,  # test cases (baseline responses) + test cases (novel responses) = total players
                "scores_baseline": scores_baseline,
                "scores_novel": scores_novel,
            },
        )
