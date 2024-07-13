"""Given some saved agents, let's them play the game.

To use the script you have to give it a path to the saved agents pickle file. The file should be a list of agents.
The script then runs the game until all agents have finished.

Example usage (with rendering):

python -m mappo_epc.replay --agents="final_agents.pkl" --name="final_rewards" --render


All parameters for the replay are:

usage: The replay script replays the pressureplate game with some provided agents. [-h] --agents AGENTS [--num_episodes NUM_EPISODES] [--max_steps MAX_STEPS] --name NAME [--render]

options:
  -h, --help            show this help message and exit
  --agents AGENTS       Path to the saved agent objects (default: None)
  --num_episodes NUM_EPISODES
                        The number of episodes to play (default: 100)
  --max_steps MAX_STEPS
                        The maximum steps per episode (default: 1000)
  --name NAME           The name you want to give to the output rewards file (default: None)
  --render              Whether to render the environment. (default: False)

"""
import argparse
import logging.config
import pickle

from datetime import datetime
from pathlib import Path

import gym
import numpy as np
import pressureplate
import torch
from torch.distributions import Categorical
import pandas as pd

Path("logs").mkdir(exist_ok=True)

logging.config.fileConfig("logging.ini")

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="The replay script replays the pressureplate game with some provided agents.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--agents', help='Path to the saved agent objects', required=True, type=str)
    parser.add_argument('--num_episodes', help='The number of episodes to play', default=100, type=int)
    parser.add_argument('--max_steps', help='The maximum steps per episode', default=1000, type=int)
    parser.add_argument('--name', help='The name you want to give to the output rewards file', required=True, type=str)
    parser.add_argument('--render', help='Whether to render the environment.', default=False, action="store_true")

    args = parser.parse_args()

    logger.info("Starting the replay.")
    logger.info(f"Arguments: {args}")

    with open(args.agents, 'rb') as f:
        agents = pickle.load(f)

    env = gym.make(f"pressureplate-linear-{len(agents)}p-v0")

    obs, *_ = env.reset()
    obs = torch.tensor(obs).unsqueeze(dim=0)

    rewards = []

    for ep in range(args.num_episodes):
        episode_rewards = []
        done = False
        num_steps = 0

        while not done and num_steps < args.max_steps:
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

            obs, reward, done, _ = env.step(actions)
            done = all(done)
            obs = torch.tensor(obs).unsqueeze(dim=0)

            episode_rewards.append(reward)
            num_steps += 1

        episode_rewards = np.array(episode_rewards)
        rewards.append(episode_rewards.mean())

    rewards = pd.DataFrame(rewards, columns=['avg_episode_reward'])

    dt_string = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    out_path = f'{args.name}_{dt_string}_rewards.csv'

    logger.info(f"Saving rewards to {out_path}")
    rewards.to_csv(out_path)
