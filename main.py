import sys
from AbstractUserContextManager import AbstractUserContextManager
from AbstractAgent import AbstractAgent
import matplotlib.pyplot as plt
import csv

# Import load_data function here


def main():

    # Command Line Arguments
    script_name = sys.argv[0]
    dataset_location = sys.argv[1]
    algorithm_name = sys.argv[2]
    time_steps = sys.argv[3]
    filename = sys.argv[4]

    # Instantiating userContextManager and agent
    userContextManager, network = load_data(dataset_location)
    agent = AbstractAgent(algorithm_name)
    
    # The list of results
    results = []

    # Percentage of optimal payoffs
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

    with open(filename, "w") as outfile:
        for num in results:
            outfile.write('{0}'.format(num))
            outfile.write("\n")

if __name__ == '__main__':
    main()
