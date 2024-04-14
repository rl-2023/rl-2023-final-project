import gym
import numpy as np

# need to run the pressure plate import to register the environment with gym
import pressureplate
import torch

from entity_encoder import EntityEncoder, ObservationEncoder, ObservationActionEncoder
from observation import Observation, get_visible_agent_observations

if __name__ == "__main__":
    num_players = 4

    # the number of grids that are contained in each observation
    num_grids = 4

    # the dimensionality of the coordinates, since we are in a 2D world, we only have x,y coordinates
    dim_coordinates = 2

    env = gym.make(f'pressureplate-linear-{num_players}p-v0')

    max_dist_visibility = env.unwrapped.max_dist
    # get the shape of the observation space from the environment
    observation_shape = env.observation_space.spaces[0].shape[0]
    observation_length = (observation_shape - dim_coordinates) // num_grids

    env.reset()

    # all agents go up
    observations, rewards, dones, _ = env.step([0, 0, 0, 0])
    observation_stack = torch.Tensor(observations)

    # TODO make classes handle the batch dimension
    oa_encoder = ObservationActionEncoder(0, observation_length, max_dist_visibility, 512)
    oa_encoder(observation_stack, 0)
