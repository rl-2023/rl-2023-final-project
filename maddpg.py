"""Contains classes and functionality related to the Reinforcement Learning part of MADDPG.

The module contains for example the Q function and policy network, which are made up of building blocks from the
encoder module. Any agent learning the pressureplate environment should only have to interact with classes and functions
from this module.
"""

import torch
from torch import nn

from encoder import ObservationActionEncoder, EntityAttention


class Q(nn.Module):
    """The Q function of the MADDPG algorithm for the pressure plate environment.

    The Q function estimates the discounted sum of expected rewards for a state-action pair. The function combines the
    embeddings of ObservationActionEncoder for each agent in the system, via an attention mechanism, and forwards them
    through a fully connected layer to get a single Q-value. Each agent has an instance of this Q function, with a
    single shared ObservationActionEncoder.

    The input to the Q function should be a tensor of shape (batch_size, environment_dim), as well as a tensor of the
    action taken by the agent owning this Q function, shape (batch_size, action_dim).

    Args:
        agent (int): The agent that owns this Q-function. Will be used to index the observation tensor.
        observation_action_encoder(ObservationActionEncoder): The ObservationActionEncoder to use.

    Example usage:
        agent = 0
        oa_encoder = ObservationActionEncoder(observation_length=25, max_dist_visibility=10, dim=256, attention_dim=128)
        q = Q(agent=0, observation_action_encoder=oa_encoder)
        observation = torch.randn((8, 4, 102))
        actions = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0] * 4]).reshape(8, -1)
        q(observation, actions)
    """

    def __init__(self, agent: int, observation_action_encoder: ObservationActionEncoder):
        super().__init__()
        self.agent = agent
        self.observation_action_encoder = observation_action_encoder
        self.attention = EntityAttention(observation_action_encoder.dim, attention_dim=128)

        self.fc_agent = nn.Linear(in_features=observation_action_encoder.dim, out_features=observation_action_encoder.dim)
        self.fc_final = nn.Linear(in_features=2 * observation_action_encoder.dim, out_features=1)

    def forward(self, observation: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Calculates the Q-value for the given observation-action pair.

        The Q-value is calculated using weighted observation action encoder embeddings. As a first step, all
        observation-action pairs are pushed through an Observation-Action Encoder, that returns an embedding of the
        observation and action for each agent. Following, we calculate attention scores between the OA embedding of the
        agent owning this Q-function and the other OA embeddings. The other embeddings are then weighed using these
        scores and summed. Finally, we concatenate this weighted sum with the agent embedding that has been forwarded
        through another fully connected layer, and feed it through a final fully connected layer that outputs the
        Q-value.

        Args:
            observation (torch.Tensor): the observation from the environment, shape (batch_size, environment_dim).
            actions (torch.Tensor): the action taken by all agents, shape (batch_size, agent, action_dim).

        Returns:
            torch.Tensor: the Q-value for the given observation-action pair.
        """
        # run each agents observation-action pair through the OA encoder
        num_agents = observation.shape[1]
        oa_embeddings = []

        for agent in range(num_agents):
            # make sure we maintain batch dimension
            action = actions[:, agent].reshape(actions.shape[0], -1)
            oa_embedding = self.observation_action_encoder.forward(agent, observation, action)

            oa_embeddings.append(oa_embedding)

        # stack all the embeddings on the agent dimension such that we have the batch dimension in front
        oa_embeddings = torch.stack(oa_embeddings, dim=1)

        # get the agent that owns this Q function from all the embeddings, unsqueeze to maintain the agent dimension
        # so that we have (batch_size, agent, embedding size)
        agent_oa_embedding = oa_embeddings[:, self.agent].unsqueeze(1)

        # get the indeces of all the other agents, makes for easier array slicing
        all_agents = torch.IntTensor([range(num_agents)])
        other_agents = all_agents[all_agents != self.agent]

        # get the embeddings of all the other agents
        other_agent_oa_embeddings = oa_embeddings[:, other_agents]

        # combine the OA embeddings using attention, again we unsqueeze because the attention class I wrote requires
        # an entity dimension that we usually calculate the attention across, here all agents can be considered the
        # same entity, so it is a dummy dimension
        attn_weights = self.attention(agent_oa_embedding.unsqueeze(1), other_agent_oa_embeddings.unsqueeze(1))

        # the attention weights come in the shape (batch_size, entity, agents), but in this case we need them as
        # (batch_size, agents, 1) so that we can easily multiply them with the OA embeddings
        attn_weights = attn_weights.permute(0, 2, 1)

        # weigh the other agent embeddings with the attention weights
        other_agent_oa_embeddings = other_agent_oa_embeddings * attn_weights

        # and sum the agent OA embeddings on the agent dimension, going from (batch_size, agent, embedding dim) to
        # (batch_size, embedding_dim)
        other_agent_oa_embeddings = torch.sum(other_agent_oa_embeddings, dim=-2)

        # push agent OA embedding through FC layer, removing the entity dimension
        agent_fc = self.fc_agent(agent_oa_embedding.squeeze(1))

        # concatenate all embeddings in a vector on the embedding dimensions, to get the final vector
        oa_embeddings_vector = torch.cat((agent_fc, other_agent_oa_embeddings), dim=-1)

        # add final FC layer that outputs Q-value
        q_value = self.fc_final(oa_embeddings_vector)

        return q_value
