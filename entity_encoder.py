"""The entity encoder module provides functionality for encoding the observations seen by the agents."""
import torch
from torch import nn

from observation import get_visible_agent_observations, extract_observation_grids


class EntityEncoder(nn.Module):
    """The entity encoder maps entity observed by agents into a higher dimensional space.

    The encoder is a two layer MLP with a default output dimension of 512 and ReLU non-linearities. The program contains
    only a single instance of this encoder per entity that is shared across agents.
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
    """The observation encoder takes an observation from the pressureplate environment as input and maps the contained entities into a higher dimension.

    The observation from the pressureplate environment contains 4 entities:

    - other agents
    - pressure plates
    - doors
    - goals

    As well as the x,y coordinates of the agent. This encoder extracts each of these entities from the observation
    tensor, and encodes them into a higher dimensional space. The encoder can handle a single observation or observations
    from multiple agents. In either case, the returned tensor will be of shape (entity, agent, embedded observation).
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
        encoded_observation.shape # (5, 4, 512)
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


class ObservationActionEncoder(nn.Module):
    """The Observation-Action Encoder creates an embedding of the observation and action of each agent.

    The observation of the agent contains all other observations of the other agents in the environment. Each agent has
    their own observation action encoder. The observation-action embedding is computed by computing attention scores between the agent
    embedding and all other agent embeddings to
    """
    pass


