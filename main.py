import gym
import numpy as np

# need to run the pressure plate import to register the environment with gym
import pressureplate
import torch

from entity_encoder import EntityEncoder


if __name__ == "__main__":
    env = gym.make('pressureplate-linear-4p-v0')

    # get the shape of the observation space from the environment
    observation_shape = env.observation_space.spaces[0].shape[0]

    env.reset()

    # all agents go up
    observations, rewards, dones, _ = env.step([0, 0, 0, 0])

    encoder = EntityEncoder(in_features=observation_shape, out_features=512)

    observations = torch.Tensor(np.array(observations))
    encoded = encoder(observations)

    # shape should be (4, 512)
    print(encoded.shape)