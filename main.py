import sys
import matplotlib.pyplot as plt
import load
import getopt
from tqdm import tqdm


def parse_command_line_args(args):
    """
    Command line options:
    -d: dataset location (included are delicious-processed, lastfm-processed, 4cliques)
    -a: algorithm name (linucb, linucbsin, goblin)
    -t: time steps (typically 10000)
    -f: output_filename (for output -- csv)
    -p: alpha value (typically 0.1)
    -c: number of clusters
    --4cliques-epsilon: 4cliques payoff noise
    --4cliques-graph-noise: 4cliques graph noise
    """
    # - further arguments
    argument_list = args[1:]
    # Default options:
    arg_options = {
        'd': "4cliques",  # dataset
        'a': "linucb",  # algorithm
        't': 10000,  # timesteps
        'f': "results.csv",  # file out
        'p': 0.1,  # alpha
        'c': None, # number of clusters
        '4cliques-epsilon': 0.1,  # 4cliques payoff noise
        '4cliques-graph-noise': 0  # 4cliques graph noise
    }
    unix_options = "d:a:t:f:p:c:"
    try:
        arguments = getopt.getopt(argument_list, unix_options, ['4cliques-epsilon=', '4cliques-graph-noise='])[0]
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(0)
    for cur_arg in arguments:
        if '-d' in cur_arg:
            arg_options['d'] = cur_arg[1].lower()
        elif '-a' in cur_arg:
            arg_options['a'] = cur_arg[1].lower()
        elif '-t' in cur_arg:
            arg_options['t'] = int(cur_arg[1])
        elif '-f' in cur_arg:
            arg_options['f'] = cur_arg[1].lower()
        elif '-p' in cur_arg:
            arg_options['p'] = float(cur_arg[1])
        elif '-c' in cur_arg:
            arg_options['c'] = int(cur_arg[1])
        elif '--4cliques-epsilon' in cur_arg:
            arg_options['4cliques-epsilon'] = float(cur_arg[1])
        elif '--4cliques-graph-noise' in cur_arg:
            arg_options['4cliques-graph-noise'] = float(cur_arg[1])
        else:
            raise Exception("Error! Argument {} not found in {}.".format(cur_arg, list(arg_options.keys())))
    return arg_options


def main():
    """
    Runs one of two multi-armed bandit learners, GOB.Lin or LinUCB, in order to attempt to learn
    and predict the preferences of users either from an existing dataset using tagged contexts
    or a virtual dataset generated at runtime, called 4CLIQUES. GOB.Lin in particular benefits from a stored
    social network recording the relationships between users, which it uses to accelerate learning about
    users.
    """
    NUM_FEATURES = 25
    # read commandline arguments, first
    full_cmd_arguments = sys.argv
    args = parse_command_line_args(full_cmd_arguments)

    # place command line arguments into corresponding variables
    dataset_location = args['d']
    algorithm_name = args['a']
    time_steps = args['t']
    output_filename = args['f']
    alpha = args['p']
    num_clusters = args['c']
    four_cliques_epsilon = args['4cliques-epsilon']
    four_cliques_graph_noise = args['4cliques-graph-noise']
    # debug string to show selected arguments
    argument_detail_string = '''
    -a (algorithm): {}
    -d (dataset/dataset location): {}
    -t (time steps): {}
    -f (output filename): {}
    -p (learning rate/alpha): {}
    -c (number of clusters): {}
    --4cliques-epsilon (payoff noise, 4cliques generated dataset): {}
    --4cliques-graph-noise (graph noise for 4cliques, determines flipped edges): {}
    '''.format(algorithm_name, dataset_location, time_steps, output_filename, alpha,
               num_clusters, four_cliques_epsilon, four_cliques_graph_noise)
    print(argument_detail_string)

    # user_context_manager provides a means of obtaining users and associated contexts to choose from for that
    # user, with the goal of choosing the most preferred context.
    # network is a representation of the social network among the users.
    user_context_manager, network, cluster_to_idx, idx_to_cluster = load.load_data(dataset_location,
                                                   four_cliques_epsilon=four_cliques_epsilon,
                                                   four_cliques_graph_noise=four_cliques_graph_noise,
                                                   num_features=NUM_FEATURES,
                                                   num_clusters=num_clusters)
    print("Loaded data.")
    if cluster_to_idx and idx_to_cluster:
        cluster_data = (cluster_to_idx, idx_to_cluster)
    else:
        cluster_data = None
    agent = load.load_agent(algorithm_name, num_features=NUM_FEATURES, alpha=alpha, graph=network,
                            cluster_data=cluster_data)
    print("Loaded agent.")

    # The list of results
    results = []
    # tqdm creates the nice progress bars!
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
