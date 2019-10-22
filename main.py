import sys
from abstractUserContextManager import abstractUserContextManager
from abstractAgent import abstractAgent

# Import load_data function here


def main():

    # Command Line Arguments
    script_name = sys.argv[0]
    dataset_location = sys.argv[1]
    algorithm_name = sys.argv[2]
    time_steps = sys.argv[3]

    # Instantiating userContextManager and agent
    userContextManager, network = load_data(dataset_location)
    agent = abstractAgent(algorithm_name)

    # Main for loop
    for step in time_steps:
        user_id, contexts = userContextManager.getUserAndContexts()
        chosen_context = agent.choose(user_id, contexts)
        payoff = userContextManager.getPayoff(user_id, chosen_context)
        agent.update(payoff)

        step += 1

if __name__ == '__main__':
    main()