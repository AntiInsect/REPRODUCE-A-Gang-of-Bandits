import sys
import getopt
from AbstractUserContextManager import AbstractUserContextManager
from AbstractAgent import AbstractAgent
import matplotlib.pyplot as plt
import csv
import load

# Import load_data function here
"""
Command line options:
-d: dataset location
-a: algorithm name
-t: time steps
-f: output_filename (for output -- csv)
-n: size of context vecotrs
-alp: alpha value
"""

def commandLine(args):
    # - further arguments
    argument_list = args[1:]
    # Default: LinUCB on lastfm-processed with:
    # 10000 timesteps, 25 contexts, aplha of 2, outputing into results.csv
    arg_options = {
        'd':"lastfm-processed",
        'a':"linucb",
        't':10000,
        'f':"results.csv",
        'n':25,
        'alp':2
    }
    unix_options = "d:a:t:f:n:alp"  
    try:  
        arguments = getopt.getopt(argument_list, unix_options)[0]
    except getopt.error as err:  
        # output error, and return with an error code
        print (str(err))
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
        if '-n' in cur_arg:
            arg_options['n'] = int(cur_arg[1])
        if '-alp' in cur_arg:
            arg_options['alp'] = int(cur_arg[1])
    return arg_options

def main():
    # read commandline arguments, first
    full_cmd_arguments = sys.argv
    args = commandLine(full_cmd_arguments)
    dataset_location = args['d']
    algorithm_name = args['a'].lower()
    time_steps = args['t']
    output_filename = args['f']
    num_contexts = args['n']
    alpha = args['alp']
    print("Running on arguments: -a %s -d %s -t %i -f %s -n %i -alp %i" \
           % (algorithm_name, dataset_location, time_steps, output_filename, num_contexts, alpha))

    # Instantiating userContextManager and agent
    UserContextManager, network = load.load_data(dataset_location, num_contexts)
    agent = load.load_agent(algorithm_name, num_features=25, alpha=alpha)
    
    # The list of results
    results = []

    # Main for loop
    for step in range(time_steps):
        user_id, contexts = UserContextManager.get_user_and_contexts()
        chosen_action, chosen_context = agent.choose(user_id, contexts, step)
        payoff = UserContextManager.get_payoff(user_id, chosen_context)
        agent.update(payoff, chosen_context, user_id)
        
        if step != 0:
            results.append(results[step-1]+payoff)
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
