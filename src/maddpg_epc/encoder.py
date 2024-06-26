"""The entity encoder module provides functionality for encoding the observations seen by the agents."""

import torch
import torch.nn.functional as F
from torch import nn

from maddpg_epc.observation import extract_observation_grids


class EntityEncoder(nn.Module):
    """The entity encoder maps entity observed by agents into a higher dimensional space.

    There are 4 entity types in the pressure plate environment:
    - plates
    - goals
    - agents
    - doors

    The encoder is a two layer MLP with a default output dimension of 512 and ReLU non-linearities. The program contains
    one instance of this encoder per entity that is shared across agents so that every agent uses the same encoder to
    encode the entities they observe.

    The input to the encoder can either be a single flat observation of shape (entity dim,) or a number of observations
    in the shape (num obs, entity dim).
    """

    def __init__(self, in_features, out_features=512):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc2 = nn.Linear(in_features=out_features, out_features=out_features)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        h1 = nn.functional.relu(self.fc1(observation))
        h2 = nn.functional.relu(self.fc2(h1))

        if h2.dim() == 1:
            h2 = h2.reshape(1, -1)
        return h2


class ObservationEncoder(nn.Module):
    """The observation encoder takes an observation from the pressureplate environment as input and maps the contained
    entities into a higher dimension.

    The observation from the pressureplate environment contains 4 entities:
    - other agents
    - pressure plates
    - doors
    - goals

    As well as the x,y coordinates of the agent. This encoder extracts each of these entities from the observation
    tensor, and encodes them into a higher dimensional space. The encoder can handle a single observation or
    observations from multiple agents. In either case, the returned tensor will be of shape
    (agent, entity, embedded observation).

    The order of entities is:

    - coordinates (this is the agent representation embedding)
    - agents (embedding of the agents grid)
    - plates (embedding of the plates grid)
    - doors (embedding of the doors grid)
    - goals (embedding of the goals grid)

    Example usage with a random observation:
        observation = torch.randn((4, 102)) # 4 agents with an observation of length 102
        encoder = ObservationEncoder(observation_length=102, dim=512)
        encoded_observation = encoder(observation)
        encoded_observation.shape # (4, 5, 512)
    """

    def __init__(self, observation_length: int, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.agent_encoder = EntityEncoder(in_features=observation_length, out_features=dim)
        self.plates_encoder = EntityEncoder(in_features=observation_length, out_features=dim)
        self.doors_encoder = EntityEncoder(in_features=observation_length, out_features=dim)
        self.goal_encoder = EntityEncoder(in_features=observation_length, out_features=dim)
        self.position_encoder = EntityEncoder(in_features=2,
                                              out_features=dim)  # this is the agent representation

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Encodes the provided observation.

        Args:
            observation (torch.Tensor): the observation to encode, should be shape (batch, agents, observations)

        Returns:
            (torch.Tensor): the encoded observations of all agents, separated by entity, shape is
            (agent, entity, embedded observation)
        """
        # TODO raise error if no batch dimension found instead of fixing it
        # if we only get a single observation, make sure we have two dimensions
        if observation.dim() == 1:
            observation = observation.reshape(1, 1, -1)

        # if we get multiple observations, we want to create matrices for each entity
        if observation.dim() == 2:
            observation = observation.unsqueeze(0)

        # the grids will have dimensions (batch, agents, grid side length)
        agent_grids, plates_grids, doors_grids, goals_grids, coordinates = extract_observation_grids(
            observation)

        # encode the different observation grids, encoder receives a stack of grids
        agent_encoded = self.agent_encoder(agent_grids)
        plates_encoded = self.plates_encoder(plates_grids)
        doors_encoded = self.doors_encoder(doors_grids)
        goals_encoded = self.goal_encoder(goals_grids)
        coordinates_encoded = self.position_encoder(coordinates)

        # we return tensor (batch, entity, agent, encoding dim)
        return torch.stack(
            (coordinates_encoded, agent_encoded, plates_encoded, doors_encoded, goals_encoded),
            dim=1)


class EntityAttention(nn.Module):
    """The Entity Attention calculates attention weights between one agents entities and other visible agents' entities.

    We are interested in the attention between the entities, because we want to know how much the agent's observation is
    related to the things that other agents are seeing. The attention is done per-entity basis, because we want to know
    how e.g. one agent's perception of the doors is related to the other agents' perception of the doors.

    Example Usage with 5 entities and 3 visible agents:
        agent_obs = torch.randn((8, 5, 1, 512))
        visible_obs = torch.randn((8, 5, 3, 512))
        attention = EntityAttention(512, 128)
        attention(agent_obs, visible_obs) # will return attention weights of shape (8, 5, 3)
    """

    def __init__(self, observation_dim: int, attention_dim: int):
        super().__init__()

        self.observation_dim = observation_dim
        self.attention_dim = attention_dim

        self.w_psi = nn.Linear(in_features=observation_dim, out_features=attention_dim)
        self.w_phi = nn.Linear(in_features=attention_dim, out_features=observation_dim)

    def _attention(self, agent_obs: torch.Tensor, visible_obs: torch.Tensor) -> torch.Tensor:
        # transpose everything but the batch dimension
        visible_obs_transposed = torch.transpose(visible_obs, 1, 2)

        # here we do the inner product of X @ W_psi @ W_phi.T @ Y, using bmm for batch matrix multiplication
        # in my understanding, beta_i_j is a scalar, so I need to place the transpose on phi and the other agent,
        # otherwise we get a matrix
        attn_inner_product = torch.bmm(self.w_phi(self.w_psi(agent_obs)), visible_obs_transposed)

        return attn_inner_product

    def forward(self, agent_observation: torch.Tensor,
                visible_observations: torch.Tensor) -> torch.Tensor:
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
        if agent_observation.dim() != 4:
            raise ValueError(
                f"Expected agent observation to be 4D tensor of shape (batch_size, entity, agent, "
                f"observation_dim), but got {visible_observations.dim()} instead.")

        if visible_observations.dim() != 4:
            raise ValueError(
                f"Expected visible observations to be 4D tensor of shape (batch_size, entity, agent, "
                f"observation_dim), but got {visible_observations.dim()} instead.")

        # for each entity, calculate the attention between the agent that owns this OA encoder, and all the other agents
        num_entities = agent_observation.shape[1]
        num_visible_agents = visible_observations.shape[2]

        betas = []
        for i in range(num_entities):
            # we calculate the attention between the agent and all other agents, hence we make entity the outer loop
            # and the visible agents the inner loop
            betas_entity = []
            for j in range(num_visible_agents):
                # make sure that the visible observations have shape (batch, agent, dim)
                beta_i_j = self._attention(agent_observation[:, i],
                                           visible_observations[:, i, j].unsqueeze(1))
                betas_entity.append(beta_i_j)

            # at this stage, betas_entity contains all non-softmaxed attention scores for a given entity
            # concatenate the list of betas together so that we have the shape (batch, entity, visible agents)
            betas_entity = torch.concat(betas_entity, dim=-1)

            betas.append(betas_entity)

        # concat all the betas on the entity dimension, because each betas_entity has the values for a single entity
        betas = torch.concat(betas, dim=1)

        # do softmax to get the alphas, since we want the softmax over the betas of ALL agents for EACH entity, we apply
        # it to the final dimension
        alphas = F.softmax(betas, dim=-1)

        return alphas


class ObservationActionEncoder(nn.Module):
    """The Observation-Action Encoder creates an embedding of the observation and action of an agent.

    The observation of the agent contains all other observations of the other agents in the environment. There is only
    a single overall instance of this encoder per Q function. The observation-action embedding is computed by computing
    attention scores between the agent embedding and all other entity and agent embeddings because we are interested in
    how the agent in question relates to all of his surroundings. The agent embedding in this case is just a higher
    dimensional representation of the agents position.

    The input to the encoder should be a tensor of shape (agents, observations) where observations are the raw
    observations of the environment.

    Args:
        observation_length (int): The length of a flattened 2D observation in the pressure plate environment.
        dim (int): the dimensionality of the final embedding.
        attention_dim (int): The dimensionality of the attention weights.
        action_in_features (int): The dimensionality of the action space, for the pressureplate environment actions are
        integers, hence the default is 1.

    Example Usage:
        oa_encoder = ObservationActionEncoder(observation_length=25, dim=256, attention_dim=128)
        agent = 0
        observation = torch.randn((8, 4, 102))
        action = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0]]).reshape(8, -1)
        oa_encoder(agent, observation, action)
    """

    def __init__(self,
                 observation_length: int,
                 dim: int = 512,
                 attention_dim: int = 128,
                 action_in_features: int = 1):
        super().__init__()
        self.dim = dim

        self.observation_encoder = ObservationEncoder(observation_length, dim)
        self.action_encoder = nn.Linear(in_features=action_in_features, out_features=dim)

        self.attention = EntityAttention(dim, attention_dim)

        self.fc_agent = nn.Linear(in_features=dim, out_features=dim)

        num_entities = 5

        # our final vector will have the entities from the agent and the visible agents (2 * num entities) as well as
        # the action embedding, all with dimensionality "dim".
        fc_final_dim_in = (2 * num_entities + 1) * dim
        self.fc_final = nn.Linear(in_features=fc_final_dim_in, out_features=dim)

    def forward(self, agent: int, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Creates the embedding of the provided observation and action.

        Args:
            agent (int): The agent for which to create the embedding.
            observation (torch.Tensor): the observation to be encoded, should be shape (batch, agents, environment_dim),
            where environment_dim is the length of the flattened 2D grid that the agent sees.
            action (int): the action that was taken by the agent that we encode the observation for.

        Returns:
            (torch.Tensor): the embedding of the observation and action.
        """
        if action.dim() != 2:
            raise ValueError(
                f"Action must have a batch dimension, found {action.dim()} dimensions.")

        action_encoded = self.action_encoder(action)

        # the observations visible to use are all the other agents' observations
        num_agents = observation.shape[1]
        other_agents = [other_agent for other_agent in range(num_agents) if other_agent != agent]
        visible_observations = observation[:, other_agents]

        # maintain the agent dimension, we always want shape (batch, agent, dim)
        agent_observations = observation[:, agent].unsqueeze(1)

        # the encoded observations are of shape (batch, entity, agent, dim)
        agent_obs_encoded = self.observation_encoder(agent_observations)
        visible_obs_encoded = self.observation_encoder(visible_observations)

        # calculate the attention weights
        alphas = self.attention(agent_obs_encoded, visible_obs_encoded)

        # finally, we use the alphas which are just importance values to weigh the observations of the agents visible
        # to the agent owning this OA, and then sum up along the agent dimension, so that we have the new embeddings
        # of the entities now
        # we have to unsqueeze the alphas so that they have shapes (batch, num_entities, num_agents, 1) and can be used
        # to weigh the observations visible to the agent
        entities_weighted = alphas.unsqueeze(-1) * visible_obs_encoded

        # sum along the agent dimension, which means that we now have a tensor of shape
        # (batch, num_entities, embedding_dim) where each row corresponds to an entity embedding combined from all
        # embeddings visible to the current agent
        summed_entities = torch.sum(entities_weighted, dim=-2)

        # remove the agent dimension from the agent obs so that we have the same number of dims as the summed entities,
        # (batch, entity, embedding dim)
        agent_obs_encoded = agent_obs_encoded.squeeze(-2)

        # we first concatenate the agent observations with the weighted entities, along the entity dimension,
        # maintaining shape (batch, entity, embedding)
        obs_concatenated = torch.cat((agent_obs_encoded, summed_entities), dim=-2)

        # then we want to flatten it into (batch, embedding) so that it can be concatenated together with the action
        # embedding
        obs_flattened = obs_concatenated.flatten(-2, -1)

        # and finally we can concatenate the observations with the action to get our final observation-action vector
        obs_action_vector = torch.concat((obs_flattened, action_encoded), dim=-1)

        return self.fc_final(obs_action_vector)
