import argparse
import torch
from maddpg_epc.train_agents import TrainingAgents
import warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser(description='MADDPG RL Parameters')

    # Environment parameters
    parser.add_argument('--num_agents', type=int, default=4, help='Number of agents')
    parser.add_argument('--episodes', type=int, default=10**1, help='Number of episodes')
    parser.add_argument('--steps_per_episode', type=int, default=10**8, help='Number of steps per episode')
    parser.add_argument('--batch_size', type=int, default=2**10, help='Batch size')
    parser.add_argument('--buffer_size', type=int, default=10**6, help='Replay buffer size')

    # Learning parameters
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.01, help='Soft update parameter')

    # Verbosity
    parser.add_argument('--verbose', action='store_true', help='Print training progress')

    return parser.parse_args()


if __name__ == '__main__':
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.__version__)
    print(device)

    args = parse_arguments()

    training_agents = TrainingAgents(num_agents=args.num_agents,
                                     episodes=args.episodes,
                                     steps_per_episode=args.steps_per_episode,
                                     batch_size=args.batch_size,
                                     buffer_size=args.buffer_size,
                                     learning_rate=args.learning_rate,
                                     gamma=args.gamma,
                                     tau=args.tau,
                                     verbose_train=args.verbose)  #.to(self.device)

    training_agents.train()



