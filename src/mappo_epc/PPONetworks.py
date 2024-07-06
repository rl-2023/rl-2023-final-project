import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .KAN import KANLayer


class Attention(nn.Module):
    """Calculate the attention weights between the focus agent observation and the other observations.

    To make the population invariant architecture work, we need to have a fixed size input to the networks. To do this,
    we want to create a weighted sum of the other observations, such that no matter how many agents we have in the game,
    their observation will always collapse to a fixed size.
    """

    def __init__(self, observation_dim: int, attention_dim: int):
        super().__init__()

        self.observation_dim = observation_dim
        self.attention_dim = attention_dim

        self.w_psi = nn.Linear(in_features=observation_dim, out_features=attention_dim)
        self.w_phi = nn.Linear(in_features=attention_dim, out_features=observation_dim)

    def _attention(self, agent_obs: torch.Tensor, visible_obs: torch.Tensor) -> torch.Tensor:
        visible_obs_transposed = torch.transpose(visible_obs, 1, 2)

        # here we do the inner product of X @ W_psi @ W_phi.T @ Y, using bmm for batch matrix multiplication
        # in my understanding, beta_i_j is a scalar, so I need to place the transpose on phi and the other agent,
        # otherwise we get a matrix
        attn_inner_product = torch.bmm(self.w_phi(self.w_psi(agent_obs)), visible_obs_transposed)

        return attn_inner_product

    def forward(self, agent_observation: torch.Tensor, visible_observations: torch.Tensor) -> torch.Tensor:
        """Calculate the entity attention scores between the agents' observation and the visible observations of other
        agents.

        Args:
            agent_observation (torch.Tensor): The agent's encoded observation of shape
            (batch_size, entity, agent, observation_dim).
            visible_observations (torch.Tensor): The agent's encoded observation of shape
            (batch_size, entity, agent, observation_dim).

        Returns:
            (torch.Tensor): The attention weights for the visible observations, in shape (batch_size, entity, agent)
        """
        # for each entity, calculate the attention between the agent that owns this OA encoder, and all the other agents
        num_visible_agents = visible_observations.shape[1]

        betas = []

        for i in range(num_visible_agents):
            # make sure that the visible observations have shape (batch, agent, dim)
            beta = self._attention(agent_observation, visible_observations[:, i].unsqueeze(1))
            betas.append(beta)

        # do softmax to get the alphas, since we want the softmax over the betas of ALL agents for EACH entity, we apply
        # it to agent dimension
        alphas = F.softmax(torch.concat(betas, dim=1), dim=1)

        # make the alpha values match the input dimension (batch, agent, dim)
        return alphas


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
            KANLayer(32, 1).to(self.device))

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
                 agent_id: int,
                 obs_space_size,
                 n_agents,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.agent_id = agent_id
        self.device = device
        self.attn = Attention(obs_space_size, 64)
        
        '''
        self.mha_v = nn.MultiheadAttention(embed_dim=obs_space_size * 2, num_heads=2)
        self.norm_v = nn.LayerNorm(obs_space_size * 2)
        self.linear_v = nn.Linear(obs_space_size * 2, 64 * 2)
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
            nn.Linear(32, 1))        
        '''
        self.value_layers = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear( obs_space_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            #nn.Linear(32,32),
            #nn.ReLU(),
            nn.Linear(32, 1))



    def value(self, obs):
        attn_obs, _ = self.mha_v(obs, obs, obs)
        z = self.activation_v(self.linear_v(self.norm_v(obs + attn_obs)))
        value = self.value_layers(z)
        return value

    def forward(self, obs):
        # get all the IDs that are not from this agent
        agent_ids = torch.tensor([i for i in range(0, obs.shape[1]) if i != self.agent_id])

        # get the obs from all the other agents
        other_obs = obs[:, agent_ids]

        # get the obs of the agent
        agent_obs = obs[:, [self.agent_id]]

        attn_weights = self.attn(agent_obs, other_obs)
        other_obs_weighted = other_obs * attn_weights
        other_obs_summed = torch.sum(other_obs_weighted, dim=1).unsqueeze(1)

        # concatenate the agent obs and the other summed obs to get a fixed size
        concat_obs = torch.concat((agent_obs, other_obs_summed), dim=-1)
        '''
        attn_obs, _ = self.mha_v(concat_obs, concat_obs, concat_obs)
        z = self.activation_v(self.linear_v(self.norm_v(concat_obs + attn_obs)))

        value = self.value_layers(z)
        return value   
        ''' 
        return self.value_layers(concat_obs)    


