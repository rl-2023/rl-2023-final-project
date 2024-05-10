import torch
from train_agents import TrainingAgents

if __name__ == '__main__':
    '''
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    '''
    training_agents = TrainingAgents(num_agents= 4,
                                    episodes= 10**2, 
                                    steps_per_episode= 10**2, 
                                    batch_size= 32, 
                                    buffer_size= 10**6, 
                                    learning_rate= 0.01, 
                                    gamma= 0.95, 
                                    tau= 0.01,
                                    verbose_train= True) #.to(self.device)

    training_agents.train()
    