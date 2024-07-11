import logging.config
import pickle
from pathlib import Path

from mappo_epc.mappo import parse_arguments
from mappo_epc import epc



Path("logs").mkdir(exist_ok=True)

logging.config.fileConfig("logging.ini")

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting EPC run.")
    args = parse_arguments()

    logger.info("\nNumber of parallel games: %s\n" +
                "Number of agents: %d\n" +
                "Number of episodes per game: %d\n" +
                "Max number of steps per episode: %d\n",
                args.parallel_games, args.num_agents, args.num_episodes, args.max_steps)

    crossover = epc.Crossover()
    mutation = epc.Mutation(num_agents=args.num_agents * 2,
                            num_episodes=args.num_episodes,
                            max_steps=args.max_steps,
                            render=args.render,
                            print_freq=args.print_freq,
                            ppo_clip_val=args.ppo_clip_val,
                            policy_lr=args.policy_lr,
                            value_lr=args.value_lr,
                            target_kl_div=args.target_kl_div,
                            max_policy_train_iters=args.max_policy_train_iters,
                            value_train_iters=args.value_train_iters)
    selection = epc.Selection(args.num_agents * 2)

    evolution = epc.Epc(args.parallel_games, [crossover, mutation, selection], args.num_agents,
                        args.num_episodes, args.max_steps, args.render, args.print_freq, args.ppo_clip_val,
                        args.policy_lr, args.value_lr, args.target_kl_div, args.max_policy_train_iters,
                        args.value_train_iters)

    final_agents = evolution.run()

    with open("final_agents.pkl", "wb") as f:
        pickle.dump(final_agents, f)
