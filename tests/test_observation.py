import torch

from observation import Observation, get_visible_agent_observations, extract_observation_grids


def test_observation():
    grids = torch.ones((4, 25)) * torch.Tensor([[1, 2, 3, 4]]).reshape(4, -1)
    grids = grids.reshape(1, -1)
    coordinates = torch.Tensor([[5, 6]])
    observations = torch.concatenate((grids, coordinates), dim=1)

    obs = Observation(observations)

    assert obs.agent_grid.size()[0] == 25
    assert (obs.agent_grid == 1).all()
    assert (obs.plates_grid == 2).all()
    assert obs.plates_grid.size()[0] == 25
    assert (obs.doors_grid == 3).all()
    assert obs.doors_grid.size()[0] == 25
    assert (obs.goals_grid == 4).all()
    assert obs.goals_grid.size()[0] == 25
    assert (obs.coordinates == coordinates).all()


def test_get_visible_agents():
    agent1_coords = torch.Tensor([[1, 1]])
    agent2_coords = torch.Tensor([[10, 10]])
    agent3_coords = torch.Tensor([[3, 4]])
    coords = torch.cat((agent1_coords, agent2_coords, agent3_coords), dim=0)

    visible_agents = get_visible_agent_observations(coords, 0, 10)

    assert len(visible_agents) == 1
    assert (visible_agents == agent3_coords).all()


def test_get_visible_agents_none_visible():
    agent1_coords = torch.Tensor([[1, 1]])
    agent2_coords = torch.Tensor([[10, 10]])
    agent3_coords = torch.Tensor([[3, 4]])
    coords = torch.cat((agent1_coords, agent2_coords, agent3_coords), dim=0)

    visible_agents = get_visible_agent_observations(coords, 0, 1)

    assert visible_agents.dim() == 3
    assert visible_agents.numel() == 0


def test_get_visible_agents_with_observation_objects():
    agent1_coords = torch.Tensor([[0, 0, 0, 0, 1, 1]])
    agent2_coords = torch.Tensor([[0, 0, 0, 0, 10, 10]])
    agent3_coords = torch.Tensor([[0, 0, 0, 0, 3, 4]])
    coords = torch.cat((agent1_coords, agent2_coords, agent3_coords), dim=0)

    visible_agents = get_visible_agent_observations(coords, 0, 1)

    assert visible_agents.dim() == 3
    assert visible_agents.numel() == 0


def test_observation_dims_no_batch():
    agent1_coords = torch.Tensor([[0, 0, 0, 0, 1, 1]])
    agent2_coords = torch.Tensor([[0, 0, 0, 0, 10, 10]])
    agent3_coords = torch.Tensor([[0, 0, 0, 0, 3, 4]])
    coords = torch.cat((agent1_coords, agent2_coords, agent3_coords), dim=0)

    agent_grid, plates_grid, doors_grid, goals_grid, coordinates = extract_observation_grids(coords)

    # make sure we have dimensions (batch, agents, obs)
    assert agent_grid.dim() == 3
    assert agent_grid.shape == (1, 3, 1)
    assert plates_grid.dim() == 3
    assert plates_grid.shape == (1, 3, 1)
    assert doors_grid.dim() == 3
    assert doors_grid.shape == (1, 3, 1)
    assert goals_grid.dim() == 3
    assert goals_grid.shape == (1, 3, 1)
    assert coordinates.dim() == 3
    assert coordinates.shape == (1, 3, 2)


def test_observation_dims_batch():
    agent1_coords = torch.Tensor([[0, 0, 0, 0, 1, 1]])
    agent2_coords = torch.Tensor([[0, 0, 0, 0, 10, 10]])
    agent3_coords = torch.Tensor([[0, 0, 0, 0, 3, 4]])
    coords = torch.stack((agent1_coords, agent2_coords, agent3_coords), dim=1)

    agent_grid, plates_grid, doors_grid, goals_grid, coordinates = extract_observation_grids(coords)

    # make sure we have dimensions (batch, agents, obs)
    assert agent_grid.dim() == 3
    assert agent_grid.shape == (1, 3, 1)
    assert plates_grid.dim() == 3
    assert plates_grid.shape == (1, 3, 1)
    assert doors_grid.dim() == 3
    assert doors_grid.shape == (1, 3, 1)
    assert goals_grid.dim() == 3
    assert goals_grid.shape == (1, 3, 1)
    assert coordinates.dim() == 3
    assert coordinates.shape == (1, 3, 2)
