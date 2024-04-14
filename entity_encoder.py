"""The entity encoder module provides functionality for encoding the observations seen by the agents."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from observation import get_visible_agent_observations, extract_observation_grids


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
    tensor, and encodes them into a higher dimensional space. The encoder can handle a single observation or observations
    from multiple agents. In either case, the returned tensor will be of shape (agent, entity, embedded observation).
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
        self.position_encoder = EntityEncoder(in_features=2, out_features=dim)  # this is the agent representation

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Encodes the provided observation.

        Args:
            observation (torch.Tensor): the observation to encode, should be shape (batch, agents, observations)

        Returns:
            (torch.Tensor): the encoded observations of all agents, separated by entity, shape is (agent, entity, embedded observation)
        """
        # if we only get a single observation, make sure we have two dimensions
        if observation.dim() == 1:
            observation = observation.reshape(1, -1)

        # if we get multiple observations, we want to create matrices for each entity
        if observation.dim() == 2:
            agent_grids = []
            plates_grids = []
            doors_grids = []
            goals_grids = []
            coordinates = []

            # for each agent observation, extract the observation grids
            for i in range(observation.shape[0]):
                agent_grid, plates_grid, doors_grid, goals_grid, coordinate = extract_observation_grids(observation[i])
                agent_grids.append(agent_grid)
                plates_grids.append(plates_grid)
                doors_grids.append(doors_grid)
                goals_grids.append(goals_grid)
                coordinates.append(coordinate)

        # encode the different observation grids, encoder receives a stack of grids
        agent_encoded = self.agent_encoder(torch.stack(agent_grids))
        plates_encoded = self.plates_encoder(torch.stack(plates_grids))
        doors_encoded = self.doors_encoder(torch.stack(doors_grids))
        goals_encoded = self.goal_encoder(torch.stack(goals_grids))
        coordinates_encoded = self.position_encoder(torch.stack(coordinates))

        return torch.stack((coordinates_encoded, agent_encoded, plates_encoded, doors_encoded, goals_encoded))


class EntityAttention(nn.Module):
    """The Entity Attention calculates attention weights between one agents entities and the other visible agents entities.

    We are interested in the attention between the entities because we want to know how much the agent's observation is
    related to the things that other agents are seeing. The attention is done per-entity basis, because we want to know
    how e.g. one agent's perception of the doors is related to the other agents' perception of the doors.

    Example Usage with 5 entities and 3 visible agents:
        agent_obs = torch.randn((8, 5, 512))
        visible_obs = torch.randn((8, 5, 3, 512))
        attention = EntityAttention(512, 128)
        attention(agent_obs, visible_obs) # will return attention weights of shape (8, 5, 3)
    """

    def __init__(self, observation_dim: int, attention_dim: int):
        super().__init__()

        self.observation_dim = observation_dim
        self.attention_dim = attention_dim

        # we should make the dimension the weights map to a parameter as well
        self.w_psi = Variable(torch.randn(observation_dim, attention_dim).type(torch.float32), requires_grad=True)
        self.w_phi = Variable(torch.randn(observation_dim, attention_dim).type(torch.float32), requires_grad=True)

    def forward(self, agent_observation: torch.Tensor, visible_observations: torch.Tensor) -> torch.Tensor:
        """Calculate the entity attention scores between the agents' observation and the visible observations of other
        agents.

        Args:
            agent_observation (torch.Tensor): The agent's encoded observation of shape (batch_size, entity, observation_dim).
            visible_observations (torch.Tensor): The agent's encoded observation of shape (batch_size, entity, agent, observation_dim).

        Returns:
            (torch.Tensor): The attention weights for the visible observations, in shape (batch_size, entity, agent)
        """
        # for each entity, calculate the attention between the agent that owns this OA encoder, and all the other agents
        num_entities = agent_observation.shape[0]
        num_visible_agents = visible_observations.shape[1]

        betas = []
        for i in range(num_entities):
            # we calculate the attention between the agent and all other agents, hence we make entity the outer loop
            # and the visible agents the inner loop
            betas_entity = []
            for j in range(num_visible_agents):
                # in my understanding, beta_i_j is a scalar, so I need to place the transpose on phi and the other agent, otherwise we get a matrix
                beta_i_j = agent_observation[i] @ self.w_psi @ self.w_phi.T @ visible_observations[i][j].unsqueeze(-2).T
                betas_entity.append(beta_i_j)
            betas.append(torch.Tensor(betas_entity))

        # stack all the betas so that each row contains all the betas for a given entity for all agents
        betas = torch.stack(betas)

        # do softmax to get the alphas, since we want this per entity, we apply it to each row
        alphas = F.softmax(betas, dim=1)

        return alphas


class ObservationActionEncoder(nn.Module):
    """The Observation-Action Encoder creates an embedding of the observation and action of an agent.

    The observation of the agent contains all other observations of the other agents in the environment. Each agent has
    their own observation action encoder. The observation-action embedding is computed by computing attention scores
    between the agent embedding and all other entity and agent embeddings because we are interested in how the agent in
    question relates to all of his surroundings. The agent embedding in this case is just a higher dimensional
    representation of the agents position.

    The input to the encoder should be a tensor of shape (agents, observations) where observations are the raw
    observations of the environment.
    """

    def __init__(self, agent: int, observation_length: int, max_dist_visibility: int, dim: int = 512):
        super().__init__()
        self.agent = agent
        self.max_dist_visibility = max_dist_visibility

        self.observation_encoder = ObservationEncoder(observation_length, dim)
        self.action_encoder = nn.Linear(in_features=1, out_features=dim)

        self.attention = EntityAttention(dim, 128)

    def forward(self, observation: torch.Tensor, action: int) -> torch.Tensor:
        """Creates the embedding of the provided observation and action.

        Args:
            observation (torch.Tensor): the observation to be encoded, should be shape (batch, agents, environment_dim)
            action (int): the action that was taken by the agent owning this class.

        Returns:
            (torch.Tensor): the embedding of the observation and action.
        """
        action_encoded = self.action_encoder(torch.FloatTensor([action]))
        visible_observations = get_visible_agent_observations(observations=observation, agent=self.agent,
                                                              sensor_range=self.max_dist_visibility)
        agent_observations = observation[self.agent]

        agent_obs_encoded = self.observation_encoder(agent_observations)
        visible_obs_encoded = self.observation_encoder(visible_observations)

        # calculate the attention weights
        alphas = self.attention(agent_obs_encoded, visible_obs_encoded)

        # finally, we use the alphas which are just importance values to weigh the observations of the agents visible
        # to the agent owning this OA, and then sum up along the agent dimension, so that we have the new embeddings
        # of the entities now
        # we have to unsqueeze the alphas so that they have shapes (num_entities, num_agents, 1) and can be used to
        entities_weighted = alphas.unsqueeze(2) * visible_obs_encoded

        # sum along the agent dimension, which means that we now have a tensor of shape (num_entities, embedding_dim)
        # where each row corresponds to an entity embedding combined from all embeddings visible to the current agent
        summed_entities = torch.sum(entities_weighted, dim=1)
        return torch.Tensor()
