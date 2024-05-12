"""Module with functionality for extracting and processing observations from the PressurePlate environment."""

import torch


class Observation:
    """Class for representing observations in the pressure plate environment.

    Takes a flat observation as returned from the pressure plate environment and extracts the layered grids and
    coordinates into 1D tensors.
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
        self.entity_observation = observation[:, :-2]


def extract_observation_grids(observation: torch.Tensor):
    """Extracts the 2D grids from the observation.

    The observation is a row tensor of chained 2D observations as well as the coordinates of the agent. This function
    splits the tensor into the flat grids and the coordinates of the agent and returns them. Assumes 4 grids in the
    observation.

    Args:
        observation (torch.Tensor): the observation from the pressure plate environment, in the shape (1, n) or (,n).

    Returns:
        the agent, plates, doors, goals grids and the coordinates, all as tensors.
    """
    # if we only have the observation of a single agent
    if observation.dim() == 1:
        observation = observation.reshape(1, 1, -1)

    # if we have multiple agent observations, make sure we have a batch dimension
    elif observation.dim() == 2:
        observation = observation.reshape(1, *observation.shape)

    # reshape so that we have the 4 grids separated
    grids = observation[:, :, :-2].reshape(observation.shape[0], observation.shape[1], 4, -1)


    agent_grid = grids[:, :, 0]
    plates_grid = grids[:, :, 1]
    doors_grid = grids[:, :, 2]
    goals_grid = grids[:, :, 3]
    coordinates = observation[:, :, -2:]

    return agent_grid, plates_grid, doors_grid, goals_grid, coordinates


def get_visible_agent_observations(observations: torch.Tensor, agent: int, sensor_range: int) -> torch.Tensor:
    """Returns the observations of the agents visible to the given agent.

    Checks which agents are visible to the provided agent and returns their observations. The observations of the agent
    itself are not returned. The x-y coordinates of the agents are assumed to be the last two elements in their
    observation. An agent is visible to another agent if their distance is <= sensor_range.

    Args:
        observations (torch.Tensor): observations of all agents in the shape (agent, observation).
        agent (int): the agent for which to return the visible agents observations.
        sensor_range (int): the distance that agents can see.

    Returns:
        torch.Tensor: the observations of the agents visible to the agent.
    """
    # if we have multiple agent observations, make sure we have a batch dimension
    if observations.dim() == 2:
        observations = observations.reshape(1, *observations.shape)

    batch_size = observations.shape[0]
    dim = observations.shape[-1]

    # we want the agent coords to have the same number of dims as the observations, (batch, agent, coords)
    agent_coords = observations[:, agent, -2:].unsqueeze(1)

    # find all the distances between the agent in question and all the other agent coordinates, using Manhattan because
    # we are in a 2D grid world where agents can only move vertically and horizontally.
    distances = torch.cdist(agent_coords, observations[:, :, -2:], p=1)

    # find the observations that are within sensor range, reshape so that we have (batch size, num agents)
    close_observations = (distances <= sensor_range).reshape(batch_size, -1)

    # remove the agent from the close observations
    close_observations[:, agent] = False
    close_observations = observations[close_observations].reshape(batch_size, -1, dim)

    return close_observations
