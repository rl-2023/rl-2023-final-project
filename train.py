import gym
import pressureplate
import torch
from torch.optim import Adam
import numpy as np
from collections import deque
import random

from encoder import ObservationActionEncoder
from maddpg import Q, PolicyNetwork
from loss import critic_loss, actor_loss, target_network_update

if __name__=='__main__':
    # Hyperparameters setup
    pressureplate
    episodes = 10
    steps_per_episode = 100
    batch_size = 8
    buffer_size = 100000
    learning_rate = 0.01
    gamma = 0.95  # discount factor
    tau=0.1 #delay factor update target network
    '''
    rendering=True
    '''
    # Environment setup
    num_agents = 4
    env = gym.make(f"pressureplate-linear-{num_agents}p-v0")
    observation_shape = env.observation_space.spaces[0].shape[0]
    num_grids = 4
    dim_coordinates = 2
    observation_length = (observation_shape - dim_coordinates) // num_grids
    #________________________________________________________________________________________________________________________________________
    # Initialize policy and Q-function for each agent
    agents = []
    for i in range(num_agents):
        oa_encoder = ObservationActionEncoder(observation_length, env.unwrapped.max_dist)
        q_function = Q(agent=i, observation_action_encoder=oa_encoder)
        policy_network = PolicyNetwork(agent=i, observation_length=observation_length, max_dist_visibility=env.unwrapped.max_dist)
        target_policy_network = PolicyNetwork(agent=i, observation_length=observation_length, max_dist_visibility=env.unwrapped.max_dist)
        target_policy_network.load_state_dict(policy_network.state_dict())
        agents.append({'q_function': q_function, 'policy_network': policy_network, 'target_policy_network': target_policy_network,
                    'optimizer': Adam(list(q_function.parameters()) + list(policy_network.parameters()), lr=learning_rate)})

    # Initialize Replay buffer
    #TODO create function to better handle buffer
    replay_buffer = deque(maxlen=buffer_size)
    #________________________________________________________________________________________________________________________________________

    # Training loop
    for episode in range(episodes):
        print(f"Episode: {episode}")
        observation = env.reset()
        observation_stack = torch.Tensor(np.array(observation)).unsqueeze(0)
        #print(f"--------observations shape {observation_stack.shape}")
        episode_rewards = np.zeros(num_agents)
        '''
        if  rendering:
            env.render(mode='rgb_array')        
        '''
        for step in range(steps_per_episode):
            print(f"  Step: {step}")
            actions = []
            for i in range(num_agents):
                actions.append(agents[i]['policy_network'].forward(observation_stack).item())

            next_observation, rewards, dones, _ = env.step(actions)
            replay_buffer.append((observation, actions, rewards, next_observation, dones))
            observation = next_observation
            observation_stack = torch.Tensor(np.array(observation)).unsqueeze(0)
            episode_rewards += rewards
            print(f"    Rewards: {rewards} | Cumulative: {np.sum(episode_rewards)}")

            # Updating (i.e. Learning)
            if len(replay_buffer) > batch_size:
                print(f"    Actions: {actions}")
                experiences = random.sample(replay_buffer, batch_size)
                #print(f"--------experiences size: {len(experiences[0])}")
                total_loss_critic=0
                total_loss_actor=0
                for i in range(num_agents):
                    #Update Q
                    agents[i]['optimizer'].zero_grad()
                    loss_critic = critic_loss(experiences, agents, i, gamma)
                    total_loss_critic+=loss_critic.item()
                    loss_critic.backward()
                    agents[i]['optimizer'].step()
                    #Update Policy
                    agents[i]['optimizer'].zero_grad()
                    loss_actor = actor_loss(experiences, agents, i)
                    total_loss_actor+=loss_actor.item()
                    loss_actor.backward()
                    agents[i]['optimizer'].step()                   

                print(f"    Critic Loss: {total_loss_critic} | Actor Loss: {total_loss_actor}")

            if all(dones):
                print(f"----ALL DONES----")
                break

            #update target policy network
            for i in range(num_agents):
                target_network_update(agents[i]['target_policy_network'],agents[i]['policy_network'], tau)

        #End episode, printing metrics
        print(f"Episode {episode} Total Reward: {np.sum(episode_rewards)}")
        print(f"---------------------------------------------------------")
    '''
    env.close()
    '''
    