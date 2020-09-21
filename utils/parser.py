import sys
import getopt


def parse_cl_args(args):
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
