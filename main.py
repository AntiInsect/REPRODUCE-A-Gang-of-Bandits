import sys

from load_data import load_data
from load_agent import load_agent

def main():

    # Command Line Arguments
    script_name = sys.argv[0]
    dataset_location = sys.argv[1]
    algorithm_name = sys.argv[2]
    time_steps = sys.argv[3]

    # Instantiating userContextManager and agent
    UserContextManager, network = load_data(dataset_location)
    agent = load_agent(algorithm_name)
    
    # The list of results
    results = []

    # Main for loop
    for step in range(int(time_steps)):
        user_id, contexts = UserContextManager.get_user_and_contexts()
        chosen_context = agent.choose(user_id, contexts)
        payoff = UserContextManager.get_payoff(user_id, chosen_context)
        agent.update(payoff)
        
        results.append(payoff)

    print(results)

if __name__ == '__main__':
    main()
