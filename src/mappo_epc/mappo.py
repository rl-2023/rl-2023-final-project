import logging

import gym
import pressureplate
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical
from dataclasses import dataclass
from copy import deepcopy
import argparse
import warnings

from torch.utils.tensorboard import SummaryWriter

# Suppress all warnings
warnings.filterwarnings("ignore")
"""
PPO original implementation from: https://colab.research.google.com/drive/1MsRlEWRAk712AQPmoM9X9E6bNeHULRDb?usp=sharing#scrollTo=J6-bk718ch2E

youtube explanation: https://www.youtube.com/watch?v=HR8kQMTO8bk
"""

logger = logging.getLogger()

tb_writer = SummaryWriter()


def parse_arguments():
    parser = argparse.ArgumentParser(description='MADDPG RL Parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Environment parameters
    parser.add_argument('--num_agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Number of steps per episode')
    parser.add_argument('--kan',
                        type=bool,
                        default=True,
                        help='Bool value if usining KAN networks for Actor and Critic')
    parser.add_argument('--render',
                        action='store_true',
                        help='Render the environment after each step')
    parser.add_argument('--print_freq', type=int, default=1, help='Print frequence wrt episodes')
    parser.add_argument('--parallel_games',
                        type=int,
                        default=3,
                        help='The number of parallel games to play for the EPC.')
    parser.add_argument('--ppo_clip_val', type=float, default=0.2, help='PPO clip value')
    parser.add_argument('--policy_lr', type=float, default=0.002, help='Learning rate for the policy network')
    parser.add_argument('--value_lr', type=float, default=0.02, help='Learning rate for the value network')
    parser.add_argument('--target_kl_div', type=float, default=0.02, help='Target KL divergence')
    parser.add_argument('--max_policy_train_iters', type=int, default=10, help='Maximum iterations for policy training')
    parser.add_argument('--value_train_iters', type=int, default=10, help='Number of iterations for value training')

    return parser.parse_args()


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
args = parse_arguments()

if args.kan == True:
    from .PPONetworks import KAN_ActorNetwork as ActorNetwork
    from .PPONetworks import KAN_CriticNetwork as CriticNetwork

else:
    from .PPONetworks import MLP_ActorNetwork as ActorNetwork
    from .PPONetworks import MLP_CriticNetwork as CriticNetwork


@dataclass
class Agent:
    id: int
    actor: ActorNetwork
    critic: CriticNetwork
    critic_old: CriticNetwork
    reward: []

    def avg_rewards(self):
        return np.mean(self.rewards)


class PPOTrainer:

    def __init__(self,
                 agent: Agent,
                 ppo_clip_val=0.2,
                 target_kl_div=0.01,
                 max_policy_train_iters=40,
                 value_train_iters=40,
                 policy_lr=3e-4,
                 value_lr=1e-2):
        self.agent_id = agent.id
        self.ac = agent.actor
        self.cr = agent.critic
        self.old_cr = agent.critic_old
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters

        self.policy_optim = optim.Adam(self.ac.parameters(), lr=policy_lr)
        self.value_optim = optim.Adam(self.cr.parameters(), lr=value_lr)

    def train_policy(self, obs, acts, old_log_probs, gaes, episode):
        loss_store = []
        for _ in range(self.max_policy_train_iters):

            self.policy_optim.zero_grad()

            new_logits = self.ac.policy(obs)
            new_logits = Categorical(logits=new_logits)
            new_log_probs = new_logits.log_prob(acts)

            policy_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = policy_ratio.clamp(1 - self.ppo_clip_val, 1 + self.ppo_clip_val)

            clipped_loss = clipped_ratio * gaes
            full_loss = policy_ratio * gaes
            policy_loss = -torch.min(full_loss, clipped_loss).mean()
            loss_store.append(policy_loss.item())

            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=10.0)
            self.policy_optim.step()

            kl_div = (old_log_probs - new_log_probs).mean()
            if kl_div >= self.target_kl_div:
                break

        tb_writer.add_scalar(f"policy/loss/agent_{self.agent_id}", np.mean(loss_store), episode)
        logger.info("Policy loss: avg %f, std %f", np.mean(loss_store), np.std(loss_store))

    def train_value(self, obs, returns, episode):

        loss_store = []
        eps = self.ppo_clip_val
        for _ in range(self.value_train_iters):
            self.value_optim.zero_grad()

            old_values = self.old_cr.value(obs)
            values = self.cr.value(obs)
            clipped_values = values.clamp(old_values - eps, old_values + eps)
            clipped_loss = (clipped_values - returns) ** 2
            full_value_loss = (returns - values) ** 2
            value_loss = torch.max(full_value_loss, clipped_loss).mean()

            loss_store.append(value_loss.item())
            value_loss.backward()
            self.old_cr.load_state_dict(self.cr.state_dict())
            torch.nn.utils.clip_grad_norm_(self.cr.parameters(), max_norm=10.0)
            self.value_optim.step()

        tb_writer.add_scalar(f"value/loss/agent_{self.agent_id}", np.mean(loss_store), episode)
        logger.info("Value loss: avg %f, std %f", np.mean(loss_store), np.std(loss_store))
        logger.info('----')


def discount_rewards(rewards, gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards) - 1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])


def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [
        rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)
    ]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas) - 1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])


def rollout(agents, env, max_steps=1000, render=False):
    logger.info("Doing rollout for %d steps", max_steps)
    train_data = [[[], [], [], [], []] for _ in range(env.n_agents)
                  ]  # obs, act, reward, values, act_log_probs
    obs, _ = env.reset()
    if render:
        env.render()
    ep_reward = np.zeros(env.n_agents)

    for _ in range(max_steps):
        obs_ = []
        act_ = []
        reward_ = []
        val_ = []
        act_log_prob_ = []

        val = agents[0].critic(torch.tensor([obs], dtype=torch.float32, device=DEVICE).view(1, -1))
        val = val.tolist()[0]

        for agent_idx in range(len(agents)):
            logits = agents[agent_idx].actor(
                torch.tensor([obs[agent_idx]], dtype=torch.float32, device=DEVICE))
            act_distribution = Categorical(logits=logits)
            act = act_distribution.sample()
            act_log_prob = act_distribution.log_prob(act).item()

            act = act.item()

            act_.append(act)
            val_.append(val[agent_idx])
            act_log_prob_.append(act_log_prob)

        next_obs, reward, done, _ = env.step(act_)

        if render:
            env.render()

        for agent_idx in range(len(agents)):

            reward_.append(reward[agent_idx])
            agents[agent_idx].reward.append(reward[agent_idx])
            obs_.append(obs[agent_idx])

            for i, item in enumerate((obs_[agent_idx], act_[agent_idx], reward_[agent_idx],
                                      val_[agent_idx], act_log_prob_[agent_idx])):
                train_data[agent_idx][i].append(item)

        obs = next_obs
        ep_reward += reward

        if all(done):
            logger.info('All dones')
            break

    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            train_data[i][j] = np.asarray(train_data[i][j])

    for agent_idx in range(len(agents)):
        train_data[agent_idx][3] = calculate_gaes(train_data[agent_idx][2],
                                                  train_data[agent_idx][3])

    return train_data, np.sum(ep_reward)


class Mappo:

    def __init__(self, num_agents,
                 num_episodes,
                 max_steps,
                 render,
                 print_freq,
                 ppo_clip_val,
                 policy_lr,
                 value_lr,
                 target_kl_div,
                 max_policy_train_iters,
                 value_train_iters):
        self.num_agents = num_agents
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.render = render
        self.print_freq = print_freq
        self.ppo_clip_val = ppo_clip_val
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters
        self.agents = []
        self.ppo_trainers = []

        self.env = gym.make(f"pressureplate-linear-{num_agents}p-v0")
        self.critic_net = CriticNetwork(self.env.observation_space[0].shape[0],
                                        self.env.n_agents).to(DEVICE)
        old_critic_net = deepcopy(self.critic_net)

        for agent_idx in range(self.num_agents):
            actor_net = ActorNetwork(self.env.observation_space[0].shape[0],
                                     self.env.action_space[0].n).to(DEVICE)

            agent = Agent(id=agent_idx, actor=actor_net, critic=self.critic_net, critic_old=old_critic_net, reward=[])
            ppo = PPOTrainer(agent=agent,
                             ppo_clip_val=self.ppo_clip_val,
                             policy_lr=self.policy_lr,
                             value_lr=self.value_lr,
                             target_kl_div=self.target_kl_div,
                             max_policy_train_iters=self.max_policy_train_iters,
                             value_train_iters=self.value_train_iters)

            self.agents.append(agent)
            self.ppo_trainers.append(ppo)

            # layout for the Tensorboard that we can group the losses on tensorboard
            layout = {
                "ABCDE": {
                    "policy loss": ["Multiline", [f"policy/loss/agent_{i}" for i in range(self.num_agents)]],
                    "value loss": ["Multiline", [f"value/loss/agent_{i}" for i in range(self.num_agents)]],
                },
            }
            tb_writer.add_custom_scalars(layout)

    def run(self):
        n_episodes = args.num_episodes
        print_freq = args.print_freq

        # Training loop
        ep_rewards = []

        for episode_idx in range(n_episodes):
            # Perform rollout
            train_data, reward = rollout(self.agents, self.env, args.max_steps, render=args.render)
            ep_rewards.append(reward)
            returns_ = []
            obs_ = []
            for agent_idx in range(len(self.agents)):
                # Shuffle
                permute_idxs = np.random.permutation(len(train_data[agent_idx][0]))

                # Policy data
                obs = torch.tensor(train_data[agent_idx][0][permute_idxs],
                                   dtype=torch.float32,
                                   device=DEVICE)
                acts = torch.tensor(train_data[agent_idx][1][permute_idxs],
                                    dtype=torch.int32,
                                    device=DEVICE)
                gaes = torch.tensor(train_data[agent_idx][3][permute_idxs],
                                    dtype=torch.float32,
                                    device=DEVICE)
                act_log_probs = torch.tensor(train_data[agent_idx][4][permute_idxs],
                                             dtype=torch.float32,
                                             device=DEVICE)

                # Value data
                returns = discount_rewards(train_data[agent_idx][2])[permute_idxs]
                returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

                returns_.append(returns)
                obs_.append(obs)

                self.ppo_trainers[agent_idx].train_policy(obs, acts, act_log_probs, gaes, episode_idx)

            returns_ = torch.stack(returns_).permute(1, 0)  # .view(len(train_data[agent_idx][0]),-1)
            obs_ = torch.stack(obs_).permute(1, 0, 2).contiguous().view(len(train_data[agent_idx][0]), -1)

            self.ppo_trainers[agent_idx].train_value(obs_, returns_, episode_idx)

            if (episode_idx + 1) % print_freq == 0:
                tb_writer.add_scalar("Average Episode Reward", np.mean(ep_rewards[-print_freq:]), episode_idx + 1)
                logger.info('Episode %d | Avg Reward %.1f', episode_idx + 1, np.mean(ep_rewards[-print_freq:]))
                logger.info('######################################################')


def main():
    args = parse_arguments()

    mappo = Mappo(num_agents=args.num_agents,
                  num_episodes=args.num_episodes,
                  max_steps=args.max_steps,
                  render=args.render,
                  print_freq=args.print_freq,
                  ppo_clip_val=args.ppo_clip_val,
                  policy_lr=args.policy_lr,
                  value_lr=args.value_lr,
                  target_kl_div=args.target_kl_div,
                  max_policy_train_iters=args.max_policy_train_iters,
                  value_train_iters=args.value_train_iters)

    mappo.run()


if __name__ == '__main__':
    main()
