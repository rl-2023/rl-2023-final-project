from entity_encoder import EntityEncoder, ObservationEncoder
import torch


def test_entity_encoder_dims():
    data = torch.randn((8, 4, 16))
    entity_encoder = EntityEncoder(in_features=16, out_features=32)

    encoded = entity_encoder(data)

    assert encoded.dim() == 3
    assert encoded.shape == (8, 4, 32)


def test_observation_encoder_dims():
    data = torch.randn((4, 18))
    observation_encoder = ObservationEncoder(observation_length=16, dim=32)

    encoded = observation_encoder(data)

    assert encoded.dim() == 3