import argparse

from maddpg_epc.train_agents import TrainingAgents


def parse_arguments():
    parser = argparse.ArgumentParser(description='MADDPG RL Parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Environment parameters
    parser.add_argument('--num_agents', type=int, default=4, help='Number of agents')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--steps_per_episode',
                        type=int,
                        default=100,
                        help='Number of steps per episode')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='Replay buffer size')

    # Learning parameters
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.01, help='Soft update parameter')

    # Verbosity
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--render', action='store_true', help='Render the environment after each step')

    return parser.parse_args()


if __name__ == '__main__':
    '''
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    '''
    args = parse_arguments()

    training_agents = TrainingAgents(num_agents=args.num_agents,
                                     episodes=args.episodes,
                                     steps_per_episode=args.steps_per_episode,
                                     batch_size=args.batch_size,
                                     buffer_size=args.buffer_size,
                                     learning_rate=args.learning_rate,
                                     gamma=args.gamma,
                                     tau=args.tau,
                                     verbose_train=args.verbose,
                                     render=args.render)  #.to(self.device)

    training_agents.train()
