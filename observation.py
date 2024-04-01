"""Module with functionality for extracting and processing observations from the PressurePlate environment."""
import torch


class Observation:
    """Class for representing observations in the pressure plate environment.

    Takes a flat observation as returned from the pressure plate environment and extracts the layered grids and coordinates
    into 1D tensors.
    """

    def __init__(self, observation: torch.Tensor):
        self._observation = observation

        # make sure we are dealing with 2 dimensional tensor
        if observation.dim() == 1:
            observation = observation.reshape(1, -1)

        grids = observation[:, :-2].reshape(4, -1)

        self.agent_grid = grids[0]
        self.plates_grid = grids[1]
        self.doors_grid = grids[2]
        self.goals_grid = grids[3]
        self.coordinates = observation[:, -2:].reshape(2)


def get_visible_agent_observations(observations: torch.Tensor, agent: int, sensor_range: int) -> torch.Tensor:
    """Returns the observations of the agents visible to the given agent.

    Checks which agents are visible to the provided agent and returns their observations. The observations of the agent
    itself are not returned. The x-y coordinates of the agents are assumed to be the last two elements in their observation.
    An agent is visible to another agent if their distance is <= sensor_range.

    Args:
        observations (torch.Tensor): observations of all agents in the shape (agent, observation).
        agent (int): the agent for which to return the visible agents observations.
        sensor_range (int): the distance that agents can see.

    Returns:
        torch.Tensor: the observations of the agents visible to the agent.
    """
    agent_coords = observations[agent, -2:].reshape(-1, 2)

    num_agents = observations.shape[0]
    other_agents = torch.IntTensor([other_agent for other_agent in range(num_agents) if other_agent != agent])
    other_agent_coords = observations[other_agents, -2:]

    distances = torch.cdist(agent_coords, observations, p=1)
    close_observations = torch.squeeze(distances <= sensor_range)

    # remove the agent from the close observations
    close_observations[agent] = False
    close_observations = observations[close_observations, :]

    return close_observations
