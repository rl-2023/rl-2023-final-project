import gym
# need to run the pressure plate import to register the environment with gym
import pressureplate
import torch

from encoder import ObservationActionEncoder
from maddpg import Q
from observation import extract_observation_grids

if __name__ == "__main__":
    pressureplate
    num_players = 4

    # the number of grids that are contained in each observation
    num_grids = 4

    # the dimensionality of the coordinates, since we are in a 2D world, we only have x,y coordinates
    dim_coordinates = 2

    env = gym.make(f"pressureplate-linear-{num_players}p-v0")

    max_dist_visibility = env.unwrapped.max_dist
    # get the shape of the observation space from the environment
    observation_shape = env.observation_space.spaces[0].shape[0]
    observation_length = (observation_shape - dim_coordinates) // num_grids

    env.reset()

    # all agents go up
    action = [0, 0, 0, 0]
    observations, rewards, dones, _ = env.step(action)
    observation_stack = torch.Tensor(observations)
    observation_stack = torch.stack((observation_stack, observation_stack))
    extract_observation_grids(observation_stack)

    oa_encoder = ObservationActionEncoder(observation_length, max_dist_visibility, 512)

    q_function = Q(agent=0, observation_action_encoder=oa_encoder)
    q_function(observation_stack, torch.Tensor([action, action]).reshape(2, -1))
    #oa_encoder(observation_stack, torch.Tensor([[0, 0]]).reshape(2, 1))
