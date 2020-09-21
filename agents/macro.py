import math

import numpy as np
from numpy.linalg import multi_dot

import scipy.sparse as sp_sparse
from scipy.linalg import fractional_matrix_power

from agents.abstract import AbstractAgent
from agents.goblin import GOBLinAgent


class MacroAgent(AbstractAgent):
    '''
    Implementation of GOBLin Block algorithm
    '''

    def __init__(self, graph, num_users, cluster_data, vector_size=25, alpha=0.1):
        if not cluster_data:
            raise Exception("No cluster data for macro algorithm")

        self.cluster_to_idx = cluster_data[0]
        self.idx_to_cluster = cluster_data[1]

        num_clusters = len(self.cluster_to_idx.keys())
        clustered_graph = np.zeros((num_clusters, num_clusters), dtype=np.float32)
        for i in range(num_users):
            for j in range(num_users):
                if graph[i][j]:
                    first_cluster = self.idx_to_cluster[i]
                    second_cluster = self.idx_to_cluster[j]
                    if first_cluster != second_cluster:
                        clustered_graph[first_cluster][second_cluster] += 1

        self.goblin_agent = GOBLinAgent(clustered_graph, num_clusters, vector_size, alpha)

    def choose(self, user, contexts, timestep):
        cluster_id = self.idx_to_cluster[user]
        return self.goblin_agent.choose(cluster_id, contexts, timestep)

    def update(self, user, context, reward):
        cluster_id = self.idx_to_cluster[user]
        self.goblin_agent.update(reward, context, cluster_id)