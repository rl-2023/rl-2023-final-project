from entity_encoder import EntityEncoder, ObservationEncoder
import torch


def test_entity_encoder_dims():
    data = torch.randn((8, 4, 16))
    entity_encoder = EntityEncoder(in_features=16, out_features=32)

    encoded = entity_encoder(data)

    assert encoded.dim() == 3
    assert encoded.shape == (8, 4, 32)


def test_observation_encoder_dims():
    num_agents = 4
    num_entities = 4
    encoding_dim = 32
    grid_length = 4
    environment_length = grid_length * num_entities + 2
    data = torch.randn((num_agents, environment_length))
    observation_encoder = ObservationEncoder(observation_length=grid_length, dim=encoding_dim)

    encoded = observation_encoder(data)

    assert encoded.dim() == 4
    assert encoded.shape == (1, num_entities + 1, num_agents, encoding_dim)
