import logging.config
from pathlib import Path

from mappo_epc.mappo import parse_arguments
from mappo_epc import epc

Path("logs").mkdir(exist_ok=True)

logging.config.fileConfig("logging.ini")

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting EPC run.")
    args = parse_arguments()

    crossover = epc.Crossover()
    mutation = epc.Mutation()
    selection = epc.Selection(args.parallel_games)

    evolution = epc.Epc(args.parallel_games, [crossover, mutation, selection], args.num_agents,
                        args.num_episodes, args.max_steps, args.render, args.print_freq)

    evolution.run()
