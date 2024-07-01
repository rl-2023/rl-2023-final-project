import gym
import pressureplate
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical
from dataclasses import dataclass
import argparse
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
"""
PPO original implementation from: https://colab.research.google.com/drive/1MsRlEWRAk712AQPmoM9X9E6bNeHULRDb?usp=sharing#scrollTo=J6-bk718ch2E

youtube explanation: https://www.youtube.com/watch?v=HR8kQMTO8bk
"""


class ActorNetwork(nn.Module):

    def __init__(self, obs_space_size, action_space_size):
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim=obs_space_size, num_heads=2)
        self.norm = nn.LayerNorm(obs_space_size)
        self.linear = nn.Linear(obs_space_size, 64 * 2)
        self.activation = nn.ReLU()

        self.policy_layers = nn.Sequential(nn.Dropout(0.5), nn.Linear(64 * 2, 64), nn.ReLU(),
                                           nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
                                           nn.Linear(32, action_space_size))

    def policy(self, obs):
        attn_obs, _ = self.mha(obs, obs, obs)
        z = self.activation(self.linear(self.norm(obs + attn_obs)))
        policy_logits = self.policy_layers(z)
        return policy_logits

    def forward(self, obs):
        attn_obs, _ = self.mha(obs, obs, obs)
        z = self.activation(self.linear(self.norm(obs + attn_obs)))
        policy_logits = self.policy_layers(z)

        return policy_logits


class CriticNetwork(nn.Module):

    def __init__(self, obs_space_size, n_agents):
        super().__init__()

        self.mha_v = nn.MultiheadAttention(embed_dim=obs_space_size * n_agents, num_heads=n_agents)
        self.norm_v = nn.LayerNorm(obs_space_size * n_agents)
        self.linear_v = nn.Linear(obs_space_size * n_agents, 64 * 2)
        self.activation_v = nn.ReLU()

        self.value_layers = nn.Sequential(nn.Dropout(0.5), nn.Linear(64 * 2, 64), nn.ReLU(),
                                          nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
                                          nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def value(self, obs):
        attn_obs, _ = self.mha_v(obs, obs, obs)
        z = self.activation_v(self.linear_v(self.norm_v(obs + attn_obs)))
        value = self.value_layers(z)
        return value

    def forward(self, obs):

        attn_obs, _ = self.mha_v(obs, obs, obs)
        z = self.activation_v(self.linear_v(self.norm_v(obs + attn_obs)))

        value = self.value_layers(z)
        return value


class PPOTrainer():

    def __init__(self,
                 ActorNetwork,
                 CriticNetwork,
                 ppo_clip_val=0.2,
                 target_kl_div=0.01,
                 max_policy_train_iters=40,
                 value_train_iters=40,
                 policy_lr=3e-4,
                 value_lr=1e-2):
        self.ac = ActorNetwork
        self.cr = CriticNetwork
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters

        policy_params = list(self.ac.mha.parameters()) + \
            list(self.ac.norm.parameters()) + \
            list(self.ac.linear.parameters()) + \
            list(self.ac.activation.parameters()) + \
            list(self.ac.policy_layers.parameters())
        self.policy_optim = optim.Adam(policy_params, lr=policy_lr)

        value_params = list(self.cr.mha_v.parameters()) + \
            list(self.cr.norm_v.parameters()) + \
            list(self.cr.linear_v.parameters()) + \
            list(self.cr.activation_v.parameters()) + \
            list(self.cr.value_layers.parameters())
        self.value_optim = optim.Adam(value_params, lr=value_lr)

    def train_policy(self, obs, acts, old_log_probs, gaes):
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
            self.policy_optim.step()

            kl_div = (old_log_probs - new_log_probs).mean()
            if kl_div >= self.target_kl_div:
                break
        print(f"Policy loss: avg {np.mean(loss_store)} std {np.std(loss_store)}")

    def train_value(self, obs, returns):
        loss_store = []
        for _ in range(self.value_train_iters):
            self.value_optim.zero_grad()

            values = self.cr.value(obs)
            value_loss = (returns - values)**2
            value_loss = value_loss.mean()
            loss_store.append(value_loss.item())
            value_loss.backward()
            self.value_optim.step()
        print(f"Value loss: avg {np.mean(loss_store)} std {np.std(loss_store)}")
        print('----')


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


@dataclass
class Agent:
    actor: ActorNetwork
    #critic: CriticNetwork
    ppo: PPOTrainer
    reward: []

    def avg_rewards(self):
        return np.mean(self.rewards)


def rollout(agents, critic_net, env, max_steps=1000, render=False):
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

        val = critic_net(torch.tensor([obs], dtype=torch.float32,
                                      device=DEVICE).view(1, -1))  #TODO fit datastructure
        val = val.item()

        for agent_idx in range(len(agents)):

            logits = agents[agent_idx].actor(
                torch.tensor([obs[agent_idx]], dtype=torch.float32, device=DEVICE))
            act_distribution = Categorical(logits=logits)
            act = act_distribution.sample()
            act_log_prob = act_distribution.log_prob(act).item()

            act = act.item()

            act_.append(act)
            val_.append(val)
            act_log_prob_.append(act_log_prob)

        next_obs, reward, done, _ = env.step(act_)
        #print(f"actions {act_}  |  rewards {reward}")

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
            print('All dones')
            break

    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            train_data[i][j] = np.asarray(train_data[i][j])

    for agent_idx in range(len(agents)):
        train_data[agent_idx][3] = calculate_gaes(train_data[agent_idx][2],
                                                  train_data[agent_idx][3])

    return train_data, np.sum(ep_reward)


def parse_arguments():
    parser = argparse.ArgumentParser(description='MADDPG RL Parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Environment parameters
    parser.add_argument('--num_agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Number of steps per episode')
    parser.add_argument('--render',
                        action='store_true',
                        help='Render the environment after each step')
    parser.add_argument('--print_freq', type=int, default=1, help='Print frequence wrt episodes')
    return parser.parse_args()


def main():
    args = parse_arguments()

    env = gym.make(f"pressureplate-linear-{args.num_agents}p-v0")

    agents = []
    critic_net = CriticNetwork(env.observation_space[0].shape[0], env.n_agents).to(DEVICE)
    for agent_idx in range(args.num_agents):

        actor_net = ActorNetwork(env.observation_space[0].shape[0],
                                 env.action_space[0].n).to(DEVICE)

        ppo_ = PPOTrainer(actor_net,
                          critic_net,
                          ppo_clip_val=0.2,
                          policy_lr=0.002,
                          value_lr=0.02,
                          target_kl_div=0.02,
                          max_policy_train_iters=10,
                          value_train_iters=10)

        agents.append(Agent(actor=actor_net, ppo=ppo_, reward=[]))

    n_episodes = args.num_episodes
    print_freq = args.print_freq

    # Training loop
    ep_rewards = []

    for episode_idx in range(n_episodes):
        # Perform rollout
        train_data, reward = rollout(agents, critic_net, env, args.max_steps, render=args.render)
        ep_rewards.append(reward)
        returns_ = []
        obs_ = []
        for agent_idx in range(len(agents)):
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
            # Train model
            agents[agent_idx].ppo.train_policy(obs, acts, act_log_probs, gaes)

        returns_ = torch.stack(returns_).view(len(train_data[agent_idx][0]), -1)
        obs_ = torch.stack(obs_).permute(1, 0, 2).contiguous().view(len(train_data[agent_idx][0]),
                                                                    -1)

        ppo_.train_value(obs_, returns_)

        if (episode_idx + 1) % print_freq == 0:

            print('Episode {} | Avg Reward {:.1f}'.format(episode_idx + 1,
                                                          np.mean(ep_rewards[-print_freq:])))
            print('######################################################')


if __name__ == '__main__':
    main()
