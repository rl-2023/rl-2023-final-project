from unittest import mock

import numpy as np

from maddpg_epc import epc


def test_crossover():
    num_agents = 4
    num_parallel_games = 3
    num_samples = 4
    agents = [[i for i in range(num_agents)]] * num_parallel_games
    crossover = epc.Crossover()

    results = crossover.run(agents, c=num_samples)

    assert len(results) == num_samples
    assert len(results[0]) == num_agents * 2
    assert np.array(results).shape == (num_samples, num_agents * 2)


def test_mutation():
    num_agents = 4
    num_parallel_games = 3
    agents = [mock.Mock() for _ in range(num_agents)]
    parallel_agents = [agents] * num_parallel_games
    mutation = epc.Mutation()
    parallel_return_mock = [mock.Mock() for i in range(num_parallel_games)]
    for agents_mock in parallel_return_mock:
        agents_mock.agents = agents
    train_agents_parallel_mock = mock.Mock(return_value=parallel_return_mock)

    with mock.patch('maddpg_epc.epc.train_agents_parallel', train_agents_parallel_mock):
        results = mutation.run(parallel_agents)

        assert len(results) == num_parallel_games
        assert np.array(results).shape == (num_parallel_games, num_agents)
        assert train_agents_parallel_mock.call_count
        assert train_agents_parallel_mock.assert_called_once()


def test_selection():
    def agent_mock_with_avg_reward(avg_reward: int):
        agent_mock = mock.Mock()
        agent_mock.avg_rewards = mock.Mock(return_value=avg_reward)
        return agent_mock

    num_survivors = 2
    agents = [[agent_mock_with_avg_reward(1), agent_mock_with_avg_reward(2)], [agent_mock_with_avg_reward(3), agent_mock_with_avg_reward(6)]]
    selection = epc.Selection(num_survivors=num_survivors)

    results = selection.run(agents)

    assert len(results) == num_survivors
    assert results == [agents[-1][-1], agents[-1][0]]
