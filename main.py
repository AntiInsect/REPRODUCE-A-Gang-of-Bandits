import sys
from AbstractUserContextManager import AbstractUserContextManager
from AbstractAgent import AbstractAgent

# Import load_data function here


def main():

    # Command Line Arguments
    script_name = sys.argv[0]
    dataset_location = sys.argv[1]
    algorithm_name = sys.argv[2]
    time_steps = int(sys.argv[3])

    # Instantiating userContextManager and agent
    userContextManager, network = load_data(dataset_location)
    agent = AbstractAgent(algorithm_name)
    
    # The list of results
    results = []

    # Main for loop
    for step in range(time_steps):
        user_id, contexts = UserContextManager.get_user_and_contexts()
        chosen_context = agent.choose(user_id, contexts)
        payoff = UserContextManager.get_payoff(user_id, chosen_context)
        agent.update(payoff)
        
        results.append(payoff)

if __name__ == '__main__':
    main()
