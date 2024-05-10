from train_agents import TrainingAgents

if __name__ == '__main__':
    training_agents = TrainingAgents(num_agents= 4,
                                    episodes= 10**1, 
                                    steps_per_episode= 10**2, 
                                    batch_size= 32, 
                                    buffer_size= 10**6, 
                                    learning_rate= 0.01, 
                                    gamma= 0.95, 
                                    tau= 0.1,
                                    verbose_train= False)
    training_agents.train()
