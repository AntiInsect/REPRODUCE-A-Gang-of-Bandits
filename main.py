import sys
from AbstractUserContextManager import AbstractUserContextManager
from AbstractAgent import AbstractAgent
import matplotlib.pyplot as plt
import csv

# Import load_data function here
"""
Command line options:
-s: script name
-d: dataset location
-a: algorithm name
-t: time steps
-f: filename (for output -- csv)
"""

def commandLine(args):
    # - further arguments
    argument_list = fullCmdArguments[1:]
    # We should replace the Nones with default options
    arg_options = {
        's':None,
        'd':None,
        'a':None,
        't':None,
        'f':None
    }
    unixOptions = "s:d:a:t:f:"  
    try:  
        arguments = getopt.getopt(argumentList, unixOptions)[0]
    except getopt.error as err:  
        # output error, and return with an error code
        print (str(err))
        sys.exit(0)
    for cur_arg in arguments:
        if '-s' in cur_arg:
            arg_options['s'] = cur_arg[1]
        if '-d' in cur_arg:
            arg_options['d'] = cur_arg[1] 
        if '-a' in cur_arg:
            arg_options['a'] = cur_arg[1]
        if '-t' in cur_arg:
            arg_options['t'] = int(cur_arg[1])
        if '-f' in cur_arg:
            arg_options['f'] = cur_arg[1]
    return arg_options


def main():

<<<<<<< HEAD
    # read commandline arguments, first
    full_cmd_arguments = sys.argv
    args = commandLine(full_cmd_arguments)
    script_name = args['s']
    dataset_location = args['d']
    algorithm_name = args['a']
    time_steps = args['t']
    output_filename = args['f']
=======
    # Command Line Arguments
    script_name = sys.argv[0]
    dataset_location = sys.argv[1]
    algorithm_name = sys.argv[2]
    time_steps = int(sys.argv[3])
>>>>>>> d70fa3f4821d1674c38a5dd0b1cfc5b337e4ed0a

    # Instantiating userContextManager and agent
    userContextManager, network = load_data(dataset_location)
    agent = AbstractAgent(algorithm_name)
    
    # The list of results
    results = []

    # Count of optimal payoffs
    num_optimal_payoffs = 0

    # Main for loop
    for step in range(time_steps):
        user_id, contexts = UserContextManager.get_user_and_contexts()
        chosen_context = agent.choose(user_id, contexts)
        payoff, is_optimal = UserContextManager.get_payoff(user_id, chosen_context)
        agent.update(payoff)
        
        results.append(payoff)
        if is_optimal:
            num_optimal_payoffs += 1
        
    # Percentage of optimal payoffs
    optimal_ratio = num_optimal_payoffs / time_steps

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
