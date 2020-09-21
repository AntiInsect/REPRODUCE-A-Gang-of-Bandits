import sys
import random

from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.loader import *
from utils.parser import *



def main(dim_feature=25, debug=False):
    '''
    Runs one of two multi-armed bandit learners, GOB.Lin or LinUCB, in order to attempt to learn
    and predict the preferences of users either from an existing dataset using tagged contexts
    or a virtual dataset generated at runtime, called 4CLIQUES. GOB.Lin in particular benefits from a stored
    social network recording the relationships between users, which it uses to accelerate learning about
    users.
    '''

    # read commandline arguments and 
    # put them into corresponding variables
    args = parse_cl_args(sys.argv)

    algorithm_name = args['a']
    dataset_location = args['d']
    time_steps = args['t']
    output_filename = args['f']
    alpha = args['p']
    num_clusters = args['c']
    four_cliques_epsilon = args['4cliques-epsilon']
    four_cliques_graph_noise = args['4cliques-graph-noise']

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

    # Load a specific dataset or directly use the artificial 4Cliques
    user_contexts, network, cluster_to_idx, idx_to_cluster = \
        load_data(dataset_location=dataset_location,
                  four_cliques_epsilon=four_cliques_epsilon,
                  four_cliques_graph_noise=four_cliques_graph_noise,
                  dim_feature=dim_feature,
                  num_clusters=num_clusters)
    # identify cluster data 
    if cluster_to_idx and idx_to_cluster:
        cluster_data = (cluster_to_idx, idx_to_cluster)
    else:
        cluster_data = None    

    # Loaded a specific agent
    agent = load_agent(
        algorithm_name=algorithm_name,
        dim_feature=dim_feature,
        alpha=alpha,
        graph=network,
        cluster_data=cluster_data)
    # the dummy agent for normalization
    agent_normalized = load_agent(
        algorithm_name='dummy',
        dim_feature=dim_feature,
        alpha=alpha,
        graph=network,
        cluster_data=cluster_data)

    # main loop 
    regrets = []
    for t in tqdm(range(time_steps)):
        # sample user
        user = user_contexts.sample_user()
        # sample contexts
        contexts = user_contexts.sample_contexts(user)
        # choose context
        chosen_context = agent.choose(user, contexts, t)
        # sample reward and normalize reward with random choice
        reward = user_contexts.sample_reward(user, chosen_context) - \
                user_contexts.sample_reward(user, agent_normalized.choose(user, contexts, t))
        # update agent
        agent.update(user, chosen_context, reward)
        # collect regrets
        if t != 0: regrets.append(regrets[t-1] + reward)
        else: regrets.append(reward)

    # plot regrets and save results
    plt.plot(regrets)
    plt.xlabel('Time')
    plt.ylabel('Cumulative reward')
    plt.show()

    with open(output_filename, "w") as outfile:
        for num in regrets:
            outfile.write('{0}'.format(num))
            outfile.write("\n")

if __name__ == '__main__':
    main()
