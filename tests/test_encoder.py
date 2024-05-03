from encoder import EntityEncoder, ObservationEncoder, EntityAttention, ObservationActionEncoder
from maddpg import Q, PolicyNetwork
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


def test_attention_dims():
    batch_size = 8
    num_agents = 4
    num_entities = 4
    agent_obs = torch.randn((batch_size, num_entities, 1, 512))
    visible_obs = torch.randn((batch_size, num_entities, num_agents, 512))
    entity_attention = EntityAttention(512, 128)

    alphas = entity_attention(agent_obs, visible_obs)

    assert alphas.dim() == 3
    assert alphas.shape == (batch_size, num_agents, num_agents)


def test_obseration_action_encoder_dims():
    batch_size = 8
    num_agents = 4
    dim = 128
    action = torch.Tensor([[1] * batch_size]).reshape(batch_size, -1)
    obs = torch.randn((batch_size, num_agents, 102))
    oa_encoder = ObservationActionEncoder(observation_length=25, max_dist_visibility=10, dim=dim)

    encoded = oa_encoder(agent=0, observation=obs, action=action)

    assert encoded.dim() == 2
    assert encoded.shape == (batch_size, dim)


def test_q_function_dims():
    batch_size = 8
    num_agents = 4
    dim = 128
    action = torch.Tensor([[1] * batch_size * num_agents]).reshape(batch_size, -1)
    obs = torch.randn((batch_size, num_agents, 102))
    oa_encoder = ObservationActionEncoder(observation_length=25, max_dist_visibility=10, dim=dim)
    q = Q(0, oa_encoder)

    q_value = q(obs, action)

    assert q_value.dim() == 2
    assert q_value.shape == (batch_size, 1)


def test_policy_network_dims():
    batch_size = 8
    num_agents = 4
    dim = 128
    obs = torch.randn((batch_size, num_agents, 102))
    policy_network = PolicyNetwork(agent=0, observation_length=25, max_dist_visibility=10, dim=dim)

    action = policy_network(observation=obs)

    assert action.dim() == 2
    assert action.shape == (batch_size, 1)
