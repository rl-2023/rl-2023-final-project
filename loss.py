import torch
import numpy as np

def critic_loss(experiences, agents, i, gamma, verbose=False):
    '''
    memo:
    Expected ACTION shape in q_function: [batch_size, num_agents, action_dim]
    Expected OBSERVATION shape in policy_network/q_function: (batch, agents, environment_dim) where environment_dim is the length of the flattened 2D grid that the agent sees.    
    '''
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

def actor_loss(experiences, agents, i, verbose=False):
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

def target_network_update(target,main, tau):
    with torch.no_grad():
        for param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data = tau * param.data + (1 - tau) * target_param.data


