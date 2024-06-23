"""Runs the evolutionary population curriculum algorithm using the maddpg algorithm."""
import abc
import concurrent.futures
import itertools
import random
from typing import Iterable

from maddpg_epc.train_agents import Agent, TrainingAgents


# train K sets of N agents (single role or as many roles as pressure plates?)
# scale each set from N to cN by copying the set, e.g. c=1.5 by the means of crossover
# Crossover
# for each set, create all the pairs
# do a union of each pair to get 2N agents
# sample C pairs
# Mutation
# for each of the C pairs, run MADDPG fine tuning
# Selection
# compute fitness score for each agent as average reward over training
# pick the top K agents

def train_agents_parallel(agents: Iterable[TrainingAgents]):
    """Runs the training for games in parallel.

    Args:
        agents (Iterable[TrainingAgents]): The games in which to train agents.

    Returns:
        Iterable[TrainingAgents]: The trained agents.
    """

    def train(train_agent: TrainingAgents):
        train_agent.train()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(train, agents))

    return results


class EvolutionaryStage(abc.ABC):
    """Represents a stage in the evolutionary population curriculum algorithm.

    Each stage operates on k parallel sets of agents and returns k parallel sets of agents.
    """

    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def run(self, population: Iterable[Iterable[Agent]], **kwargs) -> Iterable[Iterable[Agent]]:
        """Runs the stage on the population.

        Args:
            population (Iterable[Iterable[Agent]]): the population of agents.
        """
        pass


class Crossover(EvolutionaryStage):
    """The crossover stage.

    In the crossover (mix and match) stage, we scale the population by a constant factor c. The population consists of K
    parallel sets of individuals, for each of the \omega roles in the game. As a first step, we all possible set
    pairs for agents that have the same role. We do a union for each pair of sets of agents that have the same role,
    such that we now have sets of size 2N. For each role, we then pick a random set from the pairs of agents and combine
    them into another set. Finally, we sample C random pairs.
    """

    def __init__(self, name="Crossover"):
        super().__init__(name)

    def run(self, population: Iterable[Iterable[Agent]], **kwargs) -> Iterable[Iterable[Agent]]:
        """Runs the crossover stage."""
        agent_pairs = map(set, list(itertools.combinations_with_replacement(population, 2)))
        agent_pairs = [(pairs[0] | pairs[1]) for pairs in agent_pairs]

        c = kwargs.get('c')
        sampled_sets = random.sample(agent_pairs, c)

        return sampled_sets


class Mutation(EvolutionaryStage):
    """The mutation stage.

    In the mutation stage, we take the scaled population and train them using MADDPG. This can be considered as mutation
    because through MADDPG we explore different directions in the parameter space without randomly modifying the high-
    dimensional agent parameter space.
    """

    def __init__(self, name="Mutation"):
        super().__init__(name)

    def run(self, population: Iterable[Iterable[Agent]], **kwargs) -> Iterable[Iterable[Agent]]:
        # create training agent classes and assign them the agents
        num_parallel_games = len(population)
        training_agents = [TrainingAgents() for _ in range(num_parallel_games)]
        for training_agents, trained_agents in zip(training_agents, population):
            training_agents.agents = trained_agents

        mutated_agents = train_agents_parallel(training_agents)

        return [mutated_agents.agents for mutated_agents in mutated_agents]


class Selection(EvolutionaryStage):
    """The selection stage.

    In the seleciton stage, we select the top K agents as measured by their average reward during training, across all
    sets of size 2N.
    """

    def __init__(self, name="Selection"):
        super().__init__(name)

    def run(self, population: Iterable[Iterable[Agent]], **kwargs) -> Iterable[Iterable[Agent]]:
        flat_agents = [agent for game in population for agent in game]
        num_survivors = kwargs.get('num_survivors')

        agents_to_score = [(agent, agent.avg_rewards()) for agent in flat_agents]
        agents_to_score = sorted(agents_to_score, key=lambda x: x[1], reverse=True)

        survivors = agents_to_score[:num_survivors]

        return [survivor[0] for survivor in survivors]


class Epc:
    """Runs the Evolutionary Population Curriculum Algorithm."""

    def __init__(self, parallel_games: int, stages: Iterable[EvolutionaryStage]):
        self.parallel_games = parallel_games
        self.stages = stages

    def run(self) -> Iterable[Agent]:
        # as a first step, we just want to train the agents in the parallel games
        training_agents = [TrainingAgents() for _ in range(self.parallel_games)]

        training_agents = train_agents_parallel(training_agents)

        agents = [ta.agents for ta in training_agents]
        for stage in self.stages:
            agents = [stage.run(agents)]

        return agents
