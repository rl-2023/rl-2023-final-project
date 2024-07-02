import numpy as np
import torch
from torch import nn
from .KAN import KANLayer


class KAN_ActorNetwork(nn.Module):

    def __init__(self,
                 obs_space_size,
                 action_space_size,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.policy_layers = nn.Sequential(
            KANLayer(obs_space_size, 64).to(self.device), nn.LayerNorm(64),
            KANLayer(64, 32).to(self.device), nn.LayerNorm(32),
            KANLayer(32, action_space_size).to(self.device))

    def policy(self, obs):
        policy_logits = self.policy_layers(obs)
        return policy_logits

    def forward(self, obs):
        policy_logits = self.policy_layers(obs)
        return policy_logits


class KAN_CriticNetwork(nn.Module):

    def __init__(self,
                 obs_space_size,
                 n_agents,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.value_layers = nn.Sequential(
            KANLayer(obs_space_size * n_agents, 64).to(self.device), nn.LayerNorm(64),
            KANLayer(64, 32).to(self.device), nn.LayerNorm(32),
            KANLayer(32, n_agents).to(self.device))

    def value(self, obs):
        value = self.value_layers(obs)
        return value

    def forward(self, obs):
        value = self.value_layers(obs)
        return value


class MLP_ActorNetwork(nn.Module):

    def __init__(self,
                 obs_space_size,
                 action_space_size,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.mha = nn.MultiheadAttention(embed_dim=obs_space_size, num_heads=2)
        self.norm = nn.LayerNorm(obs_space_size)
        self.linear = nn.Linear(obs_space_size, 64 * 2)
        self.activation = nn.ReLU()

        self.policy_layers = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
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


class MLP_CriticNetwork(nn.Module):

    def __init__(self,
                 obs_space_size,
                 n_agents,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.mha_v = nn.MultiheadAttention(embed_dim=obs_space_size * n_agents, num_heads=n_agents)
        self.norm_v = nn.LayerNorm(obs_space_size * n_agents)
        self.linear_v = nn.Linear(obs_space_size * n_agents, 64 * 2)
        self.activation_v = nn.ReLU()

        self.value_layers = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            #nn.Linear(32,32),
            #nn.ReLU(),
            nn.Linear(32, n_agents))

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
