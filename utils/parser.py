import sys
import getopt


def parser(args):
    '''
    Command line options:
        -d: dataset location (included are delicious-processed, lastfm-processed, 4cliques)
        -a: algorithm name (linucb, linucbsin, goblin)
        -t: time steps (typically 10000)
        -f: output_filename (for output -- csv)
        -p: alpha value (typically 0.1)
        -c: number of clusters
        --4cliques-epsilon: 4cliques reward noise
        --4cliques-graph-noise: 4cliques graph noise
    '''

    # further arguments
    argument_list = args[1:]
    
    # default options:
    arg_options = {
        'd': "4cliques",  # dataset
        'a': "linucb",  # algorithm
        't': 10000,  # timesteps
        'f': "result/results.csv",  # file out
        'p': 0.1,  # alpha
        'c': None, # number of clusters
        '4cliques-epsilon': 0.1,  # 4cliques reward noise
        '4cliques-graph-noise': 0  # 4cliques graph noise
    }

    unix_options = "d:a:t:f:p:c:"
    try:
        arguments = getopt.getopt(argument_list, 
                                  unix_options,
                                  ['4cliques-epsilon=', '4cliques-graph-noise='])[0]
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(0)

    # replace default arguments with given arguments
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


def extracter(arg_options, debug=False):
    algorithm_name = arg_options['a']
    dataset_location = arg_options['d']
    time_steps = arg_options['t']
    output_filename = arg_options['f']
    alpha = arg_options['p']
    num_clusters = arg_options['c']
    four_cliques_epsilon = arg_options['4cliques-epsilon']
    four_cliques_graph_noise = arg_options['4cliques-graph-noise']

    # debug string to show selected arguments
    if debug:
        argument_detail_string = \
        '''
            -a (algorithm): {}
            -d (dataset/dataset location): {}
            -t (time steps): {}
            -f (output filename): {}
            -p (learning rate/alpha): {}
            -c (number of clusters): {}
            --4cliques-epsilon (reward noise, 4cliques generated dataset): {}
            --4cliques-graph-noise (graph noise for 4cliques, determines flipped edges): {}
        '''.format(algorithm_name,
                dataset_location,
                time_steps,
                output_filename,
                alpha,
                num_clusters,
                four_cliques_epsilon,
                four_cliques_graph_noise)
        print(argument_detail_string)
    
    return  algorithm_name, dataset_location, time_steps, output_filename, \
            alpha, num_clusters, four_cliques_epsilon, four_cliques_graph_noise