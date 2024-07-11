"""Given some saved agents, let's them play the game.

To use the script you have to give it a path to the saved agents pickle file. The file should be a list of agents.
The script then runs the game until all agents have finished.

Example usage (with rendering):

python -m mappo_epc.replay --agents="final_agents.pkl" --render
"""
import argparse
import pickle

import gym
import pressureplate
import torch
from torch.distributions import Categorical

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', help='Path to the saved agent objects', required=True, type=str)
    parser.add_argument('--render', help='Whether to render the environment.', default=False, action="store_true")

    args = parser.parse_args()

    with open(args.agents, 'rb') as f:
        agents = pickle.load(f)

    env = gym.make(f"pressureplate-linear-{len(agents)}p-v0")

    obs, *_ = env.reset()
    obs = torch.tensor(obs).unsqueeze(dim=0)

    done = False

    while not done:
        if args.render:
            env.render()

        actions = []
        for idx, agent in enumerate(agents):
            action_logits = agent.actor(obs[:, idx])

            act_distribution = Categorical(logits=action_logits)
            act = act_distribution.sample()
            act_log_prob = act_distribution.log_prob(act).item()

            act = act.item()
            actions.append(act)

        obs, _, done, _ = env.step(actions)
        done = all(done)
        obs = torch.tensor(obs).unsqueeze(dim=0)
