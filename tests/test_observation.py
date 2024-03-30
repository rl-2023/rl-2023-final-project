import torch

from observation import Observation


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
