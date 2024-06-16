"""Runs the evolutionary population curriculum algorithm using the maddpg algorithm."""
import abc
from typing import Iterable

from maddpg_epc.train_agents import Agent

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

class EvolutionaryStage(abc.ABC):
    """Represents an stage in the evolutionary population curriculum algorithm."""

    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def run(self, population: Iterable[Iterable[Agent]]):
        """Runs the stage on the population."""
        pass


class Crossover(EvolutionaryStage):
    """The crossover stage.

    In the crossover (mix and match) stage, we scale the population by a constant factor c. The population consists of K
    parallel sets of individuals, for each of the \omega roles in the game. As a first step, we all possible set
    pairs for agents that have the same role. We do a union for each pair of sets of agents that have the same role,
    such that we now have sets of size 2N. For each role, we then pick a random set from the pairs of agents and combine
    them into another set. Finally, we sample C random pairs.
    """
    def __init__(self, name = "Crossover"):
        super().__init__(name)

    def run(self, population: Iterable[Iterable[Agent]]):
        pass


class Mutation(EvolutionaryStage):
    """The mutation stage.

    In the mutation stage, we take the scaled population and train them using MADDPG. This can be considered as mutation
    because through MADDPG we explore different directions in the parameter space without randomly modifying the high-
    dimensional agent parameter space.
    """

    def __init__(self, name = "Mutation"):
        super().__init__(name)

    def run(self, population: Iterable[Iterable[Agent]]):
        pass


class Selection(EvolutionaryStage):
    """The selection stage.

    In the seleciton stage, we select the top K agents as measured by their average reward during training, across all
    sets of size 2N.
    """

    def __init__(self, name = "Selection"):
        super().__init__(name)


    def run(self, population: Iterable[Iterable[Agent]]):
        pass
