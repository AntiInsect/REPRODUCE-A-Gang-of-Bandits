import sys

from utils.loader import *
from utils.parser import *
from utils.plotter import *
from utils.runner import *


if __name__ == '__main__':
    dim_feature=25

    # Get arguments 
    args = parser(sys.argv)
    algorithm_name, dataset_location, time_steps, output_filename, \
        alpha, num_clusters, four_cliques_epsilon, four_cliques_graph_noise = extracter(arg_options=args)


    # Load a specific dataset or directly use the artificial 4Cliques
    user_contexts, network, cluster_to_idx, idx_to_cluster = \
        load_data(
            dataset_location=dataset_location,
            four_cliques_epsilon=four_cliques_epsilon,
            four_cliques_graph_noise=four_cliques_graph_noise,
            dim_feature=dim_feature,
            num_clusters=num_clusters)
    # identify cluster data 
    if cluster_to_idx and idx_to_cluster:
        cluster_data = (cluster_to_idx, idx_to_cluster)
    else:
        cluster_data = None    


    # Load a specific agent
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


    # Run the experiment 
    regrets= runner(agent=agent,
                    agent_normalized=agent_normalized,
                    user_contexts=user_contexts,
                    time_steps=time_steps)
    
    # Plot the figure and write the result
    plotter(results=regrets,
            output_filename=output_filename)
