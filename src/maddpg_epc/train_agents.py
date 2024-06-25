import random
from collections import deque
from dataclasses import dataclass

import gym
import numpy as np
import pressureplate
import torch

from maddpg_epc.encoder import ObservationActionEncoder
from maddpg_epc.maddpg import Q, PolicyNetwork
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Agent:
    q_function: Q
    policy_network: PolicyNetwork
    target_policy_network: PolicyNetwork
    optimizer_q_function: Adam
    optimizer_policy_network: Adam
    rewards: []

    def avg_rewards(self):
        return np.mean(self.rewards)


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
                 verbose_train: bool = True,
                 render: bool = False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_agents = num_agents
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.verbose_train = verbose_train
        self.render = render
        #pressureplate
        self.env = gym.make(f"pressureplate-linear-{num_agents}p-v0")
        observation_shape = self.env.observation_space.spaces[0].shape[0]
        num_grids = 4
        dim_coordinates = 2
        self.observation_length = (observation_shape - dim_coordinates) // num_grids
        
        self.agents = []
        self.initialize_agents()

        self.replay_buffer = deque(maxlen=buffer_size)
        
        # layout for the Tensorboard that we can group the losses on tensorboard
        layout = {
            "ABCDE": {
                "actor loss": ["Multiline", [f"actor/loss/agent_{i}" for i in range(self.num_agents)]],
                "critic loss": ["Multiline", [f"critic/loss/agent_{i}" for i in range(self.num_agents)]],
            },
        }
        self.tb_writer = SummaryWriter()
        self.tb_writer.add_custom_scalars(layout)

        # for keeping track of the current epoch number we train on
        self.epoch = 0

        self.total_steps = 0
        self.steps_update_interval = 100


    def initialize_agents(self):
        for i in range(self.num_agents):
            oa_encoder = ObservationActionEncoder(observation_length=self.observation_length, dim=256).to(self.device)
            q_function = Q(agent=i, observation_action_encoder=oa_encoder).to(self.device)
            policy_network = PolicyNetwork(agent=i, observation_length=self.observation_length, dim=256).to(self.device)
            target_policy_network = PolicyNetwork(agent=i, observation_length=self.observation_length, dim=256).to(self.device)
            target_policy_network.load_state_dict(policy_network.state_dict())
            self.agents.append(Agent(q_function=q_function,
                                     policy_network=policy_network,
                                     target_policy_network=target_policy_network,
                                     optimizer_q_function=Adam(list(q_function.parameters()) , lr=self.learning_rate),
                                     optimizer_policy_network=Adam( list(policy_network.parameters()), lr=self.learning_rate),
                                     rewards=[]))

    def train(self):
        for episode in range(self.episodes):
            if self.verbose_train:
                print(f"Episode: {episode}")
            observation, _ = self.env.reset()
            observation_stack = torch.Tensor(np.array(observation)).unsqueeze(0).to(self.device)
            episode_rewards = np.zeros(self.num_agents)

            for step in range(self.steps_per_episode):
                actions = [agent.policy_network.forward(observation_stack).item() for agent in self.agents]
                next_observation, rewards, dones, _ = self.env.step(actions)
                if self.render:
                    self.env.render()
                self.replay_buffer.append((observation, actions, rewards, next_observation, dones))
                observation = next_observation
                observation_stack = torch.Tensor(np.array(observation)).unsqueeze(0).to(self.device)
                episode_rewards += rewards

                # each agent tracks their own episode rewards since we need them for the selection process of the EPC
                for reward, agent in zip(rewards, self.agents):
                    agent.rewards.append(reward)

                if self.verbose_train:
                    print(f"  Step: {step}")
                    print(f"    Rewards: {rewards} | Cumulative: {np.sum(episode_rewards)}")
                    print(f"    Actions: {actions}")
                self.update_agents()

                if all(dones):
                    print(f"----ALL DONES: episode {episode} | step {step}----")
                    break

                self.total_steps += 1

            self.tb_writer.add_scalar("Total Episode Reward", np.sum(episode_rewards), episode)
            print(f"Episode {episode} | Total Reward: {np.sum(episode_rewards)}")
            print(f"---------------------------------------------------------")

    def update_agents(self):
        if len(self.replay_buffer) > self.batch_size and self.total_steps % self.steps_update_interval == 0:
            experiences = random.sample(self.replay_buffer, self.batch_size)
            total_loss_critic = 0
            total_loss_actor = 0
            for i, agent in enumerate(self.agents):
                agent.optimizer_q_function.zero_grad()
                loss_critic = self.critic_loss(experiences, self.agents, i, self.gamma).to(self.device)
                total_loss_critic += loss_critic.item()
                loss_critic.backward()
                agent.optimizer_q_function.step()

                agent.optimizer_policy_network.zero_grad()
                loss_actor = self.actor_loss(experiences, self.agents, i).to(self.device)
                total_loss_actor += loss_actor.item()
                loss_actor.backward()
                agent.optimizer_policy_network.step()

                self.tb_writer.add_scalar(f"critic/loss/agent_{i}", loss_critic, self.epoch)
                self.tb_writer.add_scalar(f"actor/loss/agent_{i}", loss_actor, self.epoch)


            for agent in self.agents:
                self.target_network_update(agent.target_policy_network, agent.policy_network, self.tau)
            if self.verbose_train:
                print(f"    Critic Loss: {total_loss_critic} | Actor Loss: {total_loss_actor}")

            # keep track of the training epochs because they are different from the steps and episodes
            self.epoch += 1

    def critic_loss(self, experiences, agents, i, gamma, verbose=False):
        """
        memo:
        Expected ACTION shape in q_function: [batch_size, num_agents, action_dim]
        Expected OBSERVATION shape in policy_network/q_function: (batch, agents, environment_dim) where environment_dim is the length of the flattened 2D grid that the agent sees.           
        """

        # Unpack experiences and networks
        observation, actions, rewards, next_observation, dones = zip(*experiences)

        num_agents=len(agents)
        q_function_i=agents[i].q_function
        target_policy_network=[{'target_policy_network':agents[j].target_policy_network} for j in range(num_agents)]

        #__________________________________________________________________________________________________
        #TODO create function for replay buffer
        # Convert to tensors with the appropriate types
        observation_stack = torch.stack([torch.Tensor(np.array(obs)).to(self.device) for obs in observation]).to(self.device)
        actions = torch.stack([torch.Tensor(np.array(act)).to(self.device) for act in actions]).unsqueeze(2).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_observation_stack = torch.stack([torch.Tensor(np.array(obs)).to(self.device) for obs in next_observation]).to(self.device)
        dones = torch.Tensor(dones).to(self.device)
        #__________________________________________________________________________________________________

        # Forward pass through the Q-function to get current Q-values
        current_q_values = q_function_i(observation_stack, actions) 
        #__________________________________________________________________________________________________

        #Computing the new actions for evaluation ot target_q_values 
        #TODO: understand if using with torch.no_grad() -> I think yes because it the target which usually is given 
        with torch.no_grad():
            next_actions=[]
            for j in range(num_agents):
                next_actions.append( self.agents[j].target_policy_network.forward(next_observation_stack).float())
            next_actions_stack=torch.stack([torch.Tensor(act).to(self.device) for act in next_actions]).permute(1, 0, 2).to(self.device)
            
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
        q_function_i=agents[i].q_function
        policy_network=[{'policy_network':agents[j].policy_network} for j in range(num_agents)]

        #__________________________________________________________________________________________________
        #TODO create function for replay buffer
        # Convert to tensors with the appropriate types
        observation_stack = torch.stack([torch.Tensor(np.array(obs)).to(self.device) for obs in observation]).to(self.device)
        #actions = torch.stack([torch.Tensor(np.array(act)) for act in actions]).unsqueeze(2)
        #rewards = torch.Tensor(rewards)
        #next_observation_stack = torch.stack([torch.Tensor(np.array(obs)) for obs in next_observation])
        #dones = torch.Tensor(dones)
        #__________________________________________________________________________________________________

        #Computing the new actions for evaluation ot target_q_values 
        #TODO: understand if using with.torch_no_grad()
        actions=[]
        for j in range(num_agents):
            actions.append( self.agents[j].policy_network.forward(observation_stack).float())
        actions_stack=torch.stack([torch.Tensor(act).to(self.device) for act in actions]).permute(1, 0, 2).to(self.device)

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

