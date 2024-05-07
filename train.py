import gym
import pressureplate
import torch
from torch.optim import Adam
import numpy as np
from collections import deque
import random

from encoder import ObservationActionEncoder
from maddpg import Q, PolicyNetwork
from observation import extract_observation_grids

def compute_loss(experiences, q_function, policy_network, gamma):
    # Unpack experiences, which include states, actions, rewards, next states, and dones
    observation, actions, rewards, next_observation, dones = zip(*experiences)

    # Convert to tensors with the appropriate types
    observation_stack = torch.stack([torch.Tensor(np.array(obs)) for obs in observation])
    print(f"observation_stack: {observation_stack.shape}")
    actions = torch.stack([torch.Tensor(np.array(act)) for act in actions]).unsqueeze(2)
    print(f"actions: {actions.shape}")
    rewards = torch.Tensor(rewards)
    next_observation_stack = torch.stack([torch.Tensor(np.array(obs)) for obs in next_observation])
    print(f"next_observation_stack: {next_observation_stack.shape}")
    dones = torch.Tensor(dones)

    # Ensure actions tensor is in the shape [batch_size, num_agents, action_dim]
    if actions.dim() == 2:  # Assume missing num_agents dimension if only 2D
        actions = actions.unsqueeze(1)  # Add num_agents dimension

    # Forward pass through the Q-function to get current Q-values
    current_q_values = q_function(observation_stack, actions).squeeze()

    # Get next actions from the policy network; detach to prevent gradients from flowing into the target
    next_actions = policy_network(next_observation_stack).unsqueeze(2).float() 
    print(f"next actions {next_actions.shape}")
    #TODO next_actions have dim (batch,1) should be (batch,agent)
    # Forward pass through the Q-function for next Q-values using the target policy's actions
    next_q_values = q_function(next_observation_stack, next_actions).squeeze()
    print(f"next_q_values {next_q_values.shape}")
    # Compute the target Q-values: reward + gamma * max(next_q_values) * (1 - dones)
    target_q_values = rewards + gamma * next_q_values * (1-dones)

    # Compute the loss using Mean Squared Error between current and target Q-values
    loss = torch.mean((current_q_values - target_q_values) ** 2)

    return loss


if __name__=='__main__':
    # Hyperparameters
    pressureplate
    episodes = 10
    steps_per_episode = 200
    batch_size = 64
    buffer_size = 10000
    learning_rate = 0.01
    gamma = 0.95  # discount factor

    # Environment setup
    
    num_agents = 4
    env = gym.make(f"pressureplate-linear-{num_agents}p-v0")
    observation_shape = env.observation_space.spaces[0].shape[0]
    num_grids = 4
    dim_coordinates = 2
    observation_length = (observation_shape - dim_coordinates) // num_grids

    # Initialize policy and Q-function for each agent
    agents = []
    for i in range(num_agents):
        oa_encoder = ObservationActionEncoder(observation_length, env.unwrapped.max_dist)
        q_function = Q(agent=i, observation_action_encoder=oa_encoder)
        policy_network = PolicyNetwork(agent=i, observation_length=observation_length, max_dist_visibility=env.unwrapped.max_dist)
        agents.append({'q_function': q_function, 'policy_network': policy_network,
                    'optimizer': Adam(list(q_function.parameters()) + list(policy_network.parameters()), lr=learning_rate)})

    # Replay buffer
    replay_buffer = deque(maxlen=buffer_size)

    '''
    observations, rewards, dones, _ = env.step(action)
    observation_stack = torch.Tensor(observations)
    observation_stack = torch.stack((observation_stack, observation_stack))
    extract_observation_grids(observation_stack)

    oa_encoder = ObservationActionEncoder(observation_length, max_dist_visibility, 512)

    q_function = Q(agent=0, observation_action_encoder=oa_encoder)
    q_function(observation_stack, torch.Tensor([action, action]).reshape(2, -1))
    '''
    # Training loop
    for episode in range(episodes):
        observation = env.reset()
        observation_stack = torch.Tensor(np.array(observation)).unsqueeze(0)
        print(f"observations shape {observation_stack.shape}")
        episode_rewards = np.zeros(num_agents)

        for step in range(steps_per_episode):
            actions = []
            for i in range(num_agents):
                # Assuming the policy network can handle the current state and output an action
                actions.append(agents[i]['policy_network'].forward(observation_stack).item())

            print(f"Actions: {actions}")
            next_observation, rewards, dones, _ = env.step(actions)
            replay_buffer.append((observation, actions, rewards, next_observation, dones))
            observation = next_observation
            episode_rewards += rewards
            print(f"Episode {episode} reward {np.sum(episode_rewards)}")

            # Learning
            if len(replay_buffer) >= batch_size:
                experiences = random.sample(replay_buffer, batch_size)
                for i in range(num_agents):
                    agents[i]['optimizer'].zero_grad()
                    loss = compute_loss(experiences, agents[i]['q_function'], agents[i]['policy_network'], gamma)
                    loss.backward()
                    agents[i]['optimizer'].step()

            if all(dones):
                break

        print(f"Episode {episode} Reward: {np.sum(episode_rewards)}")

