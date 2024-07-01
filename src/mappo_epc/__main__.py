from mappo_epc.mappo import parse_arguments
from mappo_epc import epc

if __name__ == '__main__':
    args = parse_arguments()

    crossover = epc.Crossover()
    mutation = epc.Mutation()
    selection = epc.Selection(args.parallel_games)

    evolution = epc.Epc(args.parallel_games,
                        [crossover, mutation, selection],
                        args.num_agents,
                        args.num_episodes,
                        args.max_steps,
                        args.render,
                        args.print_freq)

    evolution.run()
