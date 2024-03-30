"""The entity encoder module provides functionality for encoding the observations seen by the agents."""
import torch
from torch import nn


class EntityEncoder(nn.Module):
    """The entity encoder maps agent observations into a higher dimensional space.

    The encoder is a two layer MLP with a default output dimension of 512 and ReLU non-linearities.
    """

    def __init__(self, in_features, out_features=512):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc2 = nn.Linear(in_features=out_features, out_features=out_features)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        h1 = nn.functional.relu(self.fc1(observation))
        h2 = nn.functional.relu(self.fc2(h1))

        return h2
