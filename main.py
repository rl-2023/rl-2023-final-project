import gym
import numpy as np

# need to run the pressure plate import to register the environment with gym
import pressureplate
import torch

from entity_encoder import EntityEncoder
from observation import Observation

if __name__ == "__main__":
    num_players = 4

    # the number of grids that are contained in each observation
    num_grids = 4

    # the dimensionality of the coordinates, since we are in a 2D world, we only have x,y coordinates
    dim_coordinates = 2

    env = gym.make(f'pressureplate-linear-{num_players}p-v0')

    # get the shape of the observation space from the environment
    observation_shape = env.observation_space.spaces[0].shape[0]
    observation_length = (observation_shape - dim_coordinates) // num_grids

    env.reset()

    # all agents go up
    observations, rewards, dones, _ = env.step([0, 0, 0, 0])
    observations = [Observation(torch.Tensor(np.array(observation))) for observation in observations]

    # create one encoder for each entity in the environment
    agent_encoder = EntityEncoder(in_features=observation_length, out_features=512)
    plates_encoder = EntityEncoder(in_features=observation_length, out_features=512)
    doors_encoder = EntityEncoder(in_features=observation_length, out_features=512)
    goal_encoder = EntityEncoder(in_features=observation_length, out_features=512)

    # shape should be 512
    print(agent_encoder(observations[0].agent_grid).shape)

    # TODO for each agent, find the agents that are visible to them
    # TODO for each agent, calculate the attention between its observation/coordinates (what exactly?) and the other agent observations