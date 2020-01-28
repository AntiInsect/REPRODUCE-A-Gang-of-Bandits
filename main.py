import sys
import matplotlib.pyplot as plt
import load
import getopt
from tqdm import tqdm

# Import load_data function here
"""
Command line options:
-d: dataset location
-a: algorithm name
-t: time steps
-f: output_filename (for output -- csv)
-n: size of context vecotrs
-p: alpha value
"""


def parse_command_line_args(args):
    # - further arguments
    argument_list = args[1:]
    # Default: LinUCB on lastfm-processed with:
    # 10000 timesteps, 25 contexts, aplha of 2, outputing into results.csv
    arg_options = {
        'd': "lastfm-processed",
        'a': "linucb",
        't': 10000,
        'f': "results.csv",
        'p': 2,
        '4cliques-epsilon': 0.1,
        '4cliques-graph-noise': 0
    }
    unix_options = "d:a:t:f:n:p:"
    try:
        arguments = getopt.getopt(argument_list, unix_options, ['4cliques-epsilon=', '4cliques-graph-noise='])[0]
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(0)
    for cur_arg in arguments:
        if '-d' in cur_arg:
            arg_options['d'] = cur_arg[1].lower()
        if '-a' in cur_arg:
            arg_options['a'] = cur_arg[1].lower()
        if '-t' in cur_arg:
            arg_options['t'] = int(cur_arg[1])
        if '-f' in cur_arg:
            arg_options['f'] = cur_arg[1].lower()
        if '-p' in cur_arg:
            arg_options['p'] = float(cur_arg[1])
        if '--4cliques-epsilon' in cur_arg:
            arg_options['4cliques-epsilon'] = float(cur_arg[1])
        if '--4cliques-graph-noise' in cur_arg:
            arg_options['4cliques-graph-noise'] = float(cur_arg[1])
    return arg_options


def main():
    # read commandline arguments, first
    full_cmd_arguments = sys.argv
    args = parse_command_line_args(full_cmd_arguments)
    dataset_location = args['d']
    algorithm_name = args['a'].lower()
    time_steps = args['t']
    output_filename = args['f']
    alpha = args['p']
    four_cliques_epsilon = args['4cliques-epsilon']
    four_cliques_graph_noise = args['4cliques-graph-noise']
    argument_detail_string = '''
    -a (algorithm): {}
    -d (dataset/dataset location): {}
    -t (time steps): {}
    -f (output filename): {}
    -p (learning rate/alpha): {}
    --4cliques-epsilon (payoff noise, 4cliques generated dataset): {}
    --4cliques-graph-noise (graph noise for 4cliques, determines flipped edges): {}
    '''.format(algorithm_name, dataset_location, time_steps, output_filename, alpha,
               four_cliques_epsilon, four_cliques_graph_noise)
    print(argument_detail_string)

    # Instantiating userContextManager and agent
    user_context_manager, network = load.load_data(dataset_location,
                                                   four_cliques_epsilon=four_cliques_epsilon,
                                                   four_cliques_graph_noise=four_cliques_graph_noise)
    print("Loaded data.")
    agent = load.load_agent(algorithm_name, num_features=25, alpha=alpha, graph=network)
    print("Loaded agent.")

    # The list of results
    results = []

    for step in tqdm(range(time_steps)):
        user_id, contexts = user_context_manager.get_user_and_contexts()
        chosen_context = agent.choose(user_id, contexts, step)
        payoff = user_context_manager.get_payoff(user_id, chosen_context)
        agent.update(payoff, chosen_context, user_id)
        if step != 0:
            results.append(results[step - 1] + payoff)
        else:
            results.append(payoff)

    # Two options for data visualization:
    # Matplotlib (immediate visualization) and csv export (for later use)
    plt.plot(results)
    plt.ylabel('Cumulative payoff')
    plt.show()

    with open(output_filename, "w") as outfile:
        for num in results:
            outfile.write('{0}'.format(num))
            outfile.write("\n")


if __name__ == '__main__':
    main()
