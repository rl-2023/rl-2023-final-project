import gym
import pressureplate
import torch
from torch.optim import Adam
import numpy as np
from collections import deque
import random

from encoder import ObservationActionEncoder
from maddpg import Q, PolicyNetwork

class TrainingAgents:
    """
    Class to implement the training of a set of agents
    """
    def __init__(self, 
                 num_agents: int = 4,
                 episodes: int = 10**2, 
                 steps_per_episode: int = 10**3, 
                 batch_size: int = 256, 
                 buffer_size: int = 10**6, 
                 learning_rate: float = 0.01, 
                 gamma: float = 0.95, 
                 tau: float = 0.1,
                 verbose_train: bool = True):
        
        self.num_agents = num_agents
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.verbose_train=verbose_train
        #pressureplate
        self.env = gym.make(f"pressureplate-linear-{num_agents}p-v0")
        observation_shape = self.env.observation_space.spaces[0].shape[0]
        num_grids = 4
        dim_coordinates = 2
        self.observation_length = (observation_shape - dim_coordinates) // num_grids
        
        self.agents = []
        self.initialize_agents()

        self.replay_buffer = deque(maxlen=buffer_size)
        
    def initialize_agents(self):
        for i in range(self.num_agents):
            oa_encoder = ObservationActionEncoder(self.observation_length, self.env.unwrapped.max_dist)
            q_function = Q(agent=i, observation_action_encoder=oa_encoder)
            policy_network = PolicyNetwork(agent=i, observation_length=self.observation_length, max_dist_visibility=self.env.unwrapped.max_dist)
            target_policy_network = PolicyNetwork(agent=i, observation_length=self.observation_length, max_dist_visibility=self.env.unwrapped.max_dist)
            target_policy_network.load_state_dict(policy_network.state_dict())
            self.agents.append({
                'q_function': q_function, 
                'policy_network': policy_network, 
                'target_policy_network': target_policy_network,
                'optimizer': Adam(list(q_function.parameters()) + list(policy_network.parameters()), lr=self.learning_rate)
            })

    def train(self):
        for episode in range(self.episodes):
            if self.verbose_train:
                print(f"Episode: {episode}")
            observation = self.env.reset()
            observation_stack = torch.Tensor(np.array(observation)).unsqueeze(0)
            episode_rewards = np.zeros(self.num_agents)

            for step in range(self.steps_per_episode):
                actions = [agent['policy_network'].forward(observation_stack).item() for agent in self.agents]
                next_observation, rewards, dones, _ = self.env.step(actions)
                self.replay_buffer.append((observation, actions, rewards, next_observation, dones))
                observation = next_observation
                observation_stack = torch.Tensor(np.array(observation)).unsqueeze(0)
                episode_rewards += rewards
                if self.verbose_train:
                    print(f"  Step: {step}")
                    print(f"    Rewards: {rewards} | Cumulative: {np.sum(episode_rewards)}")
                    print(f"    Actions: {actions}")
                self.update_agents()

                if all(dones):
                    print(f"----ALL DONES: episode {episode} | step {step}----")
                    break

            print(f"Episode {episode} | Total Reward: {np.sum(episode_rewards)}")
            print(f"---------------------------------------------------------")

    def update_agents(self):
        if len(self.replay_buffer) > self.batch_size:
            experiences = random.sample(self.replay_buffer, self.batch_size)
            total_loss_critic = 0
            total_loss_actor = 0
            for i, agent in enumerate(self.agents):
                agent['optimizer'].zero_grad()
                loss_critic = self.critic_loss(experiences, self.agents, i, self.gamma)
                total_loss_critic += loss_critic.item()
                loss_critic.backward()
                agent['optimizer'].step()

                agent['optimizer'].zero_grad()
                loss_actor = self.actor_loss(experiences, self.agents, i)
                total_loss_actor += loss_actor.item()
                loss_actor.backward()
                agent['optimizer'].step()

            for agent in self.agents:
                self.target_network_update(agent['target_policy_network'], agent['policy_network'], self.tau)
            if self.verbose_train:
                print(f"    Critic Loss: {total_loss_critic} | Actor Loss: {total_loss_actor}")

    def critic_loss(self, experiences, agents, i, gamma, verbose=False):
        """
        memo:
        Expected ACTION shape in q_function: [batch_size, num_agents, action_dim]
        Expected OBSERVATION shape in policy_network/q_function: (batch, agents, environment_dim) where environment_dim is the length of the flattened 2D grid that the agent sees.           
        """

        # Unpack experiences and networks
        observation, actions, rewards, next_observation, dones = zip(*experiences)

        num_agents=len(agents)
        q_function_i=agents[i]['q_function']
        target_policy_network=[{'target_policy_network':agents[j]['target_policy_network']} for j in range(num_agents)]

        #__________________________________________________________________________________________________
        #TODO create function for replay buffer
        # Convert to tensors with the appropriate types
        observation_stack = torch.stack([torch.Tensor(np.array(obs)) for obs in observation])
        actions = torch.stack([torch.Tensor(np.array(act)) for act in actions]).unsqueeze(2)
        rewards = torch.Tensor(rewards)
        next_observation_stack = torch.stack([torch.Tensor(np.array(obs)) for obs in next_observation])
        dones = torch.Tensor(dones)
        #__________________________________________________________________________________________________

        # Forward pass through the Q-function to get current Q-values
        current_q_values = q_function_i(observation_stack, actions) 
        #__________________________________________________________________________________________________

        #Computing the new actions for evaluation ot target_q_values 
        #TODO: understand if using with torch.no_grad() -> I think yes because it the target which usually is given 
        with torch.no_grad():
            next_actions=[]
            for j in range(num_agents):
                next_actions.append( target_policy_network[j]['target_policy_network'].forward(next_observation_stack).float())
            next_actions_stack=torch.stack([torch.Tensor(np.array(act)) for act in next_actions]).permute(1, 0, 2)
            
            # Compute the target Q-values
            #TODO check if correct implementation
            target_q_values = rewards[:, i].unsqueeze(1) + gamma * q_function_i(next_observation_stack, next_actions_stack)  * (1-dones[:, i].unsqueeze(1)) 
            
        # Compute the loss using Mean Squared Error between current and target Q-values
        loss_i = torch.mean((current_q_values - target_q_values) ** 2)

        #__________________________________________________________________________________________________
        if verbose:
            print(f"--------observation_stack: {observation_stack.shape}")   
            print(f"--------actions: {actions.shape}")    
            print(f"--------rewards: {rewards.shape}")
            print(f"--------next_observation_stack: {next_observation_stack.shape}")    
            print(f"--------dones: {dones.shape}")
            print(f"--------current_q_values: {current_q_values.shape}")
            print(f"--------next actions_stack: {next_actions_stack.shape}")
            print(f"--------target_q_values: {target_q_values.shape}")

        return loss_i

    def actor_loss(self, experiences, agents, i, verbose=False):
        '''
        memo:
        Expected ACTION shape in q_function: [batch_size, num_agents, action_dim]
        Expected OBSERVATION shape in policy_network/q_function: (batch, agents, environment_dim) where environment_dim is the length of the flattened 2D grid that the agent sees.    
        '''
        # Unpack experiences and networks
        observation, actions_, rewards_, next_observation_, dones_ = zip(*experiences)

        num_agents=len(agents)
        q_function_i=agents[i]['q_function']
        policy_network=[{'policy_network':agents[j]['policy_network']} for j in range(num_agents)]

        #__________________________________________________________________________________________________
        #TODO create function for replay buffer
        # Convert to tensors with the appropriate types
        observation_stack = torch.stack([torch.Tensor(np.array(obs)) for obs in observation])
        #actions = torch.stack([torch.Tensor(np.array(act)) for act in actions]).unsqueeze(2)
        #rewards = torch.Tensor(rewards)
        #next_observation_stack = torch.stack([torch.Tensor(np.array(obs)) for obs in next_observation])
        #dones = torch.Tensor(dones)
        #__________________________________________________________________________________________________

        #Computing the new actions for evaluation ot target_q_values 
        #TODO: understand if using with.torch_no_grad()
        actions=[]
        for j in range(num_agents):
            actions.append( policy_network[j]['policy_network'].forward(observation_stack).float())
        actions_stack=torch.stack([torch.Tensor(np.array(act)) for act in actions]).permute(1, 0, 2)

        #__________________________________________________________________________________________________
        #Computing the loss which have the minus because we want gradient ascent
        actor_loss_i= -torch.mean(q_function_i(observation_stack, actions_stack)) 
        #__________________________________________________________________________________________________
        if verbose:
            print(f"--------observation_stack: {observation_stack.shape}")   
            print(f"--------actions_stack: {actions_stack.shape}")

        return actor_loss_i

    def target_network_update(self, target, main, tau):
        with torch.no_grad():
            for param, target_param in zip(main.parameters(), target.parameters()):
                target_param.data = tau * param.data + (1 - tau) * target_param.data
