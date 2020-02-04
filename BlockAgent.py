from AbstractAgent import AbstractAgent
import numpy as np
import scipy.sparse as sp_sparse
from scipy.linalg import fractional_matrix_power
from numpy.linalg import multi_dot
import math
from collections import defaultdict


class BlockAgent(AbstractAgent):
    """
    Implementation of GOBLin Block algorithm
    """

    class ClusterToMatrix:
        """
        Maintains matrix for each cluster
        """
        def __init__(self, vector_size, users, graph):
            self.num_users = len(users)

            # create graph for this cluster -- new adjacency matrix

            self.m = np.identity(self.num_users * vector_size, dtype=np.float32)
            i_n = np.identity(self.num_users, dtype=np.float32)
            # construct a laplacian matrix based on the graph, that we will modify and then
            # take the kronecker product of to get a representation of the graph that helps us learn
            # although the laplacian is sp_sparse, if called on a dense matrix it will return a dense matrix
            laplacian = sp_sparse.csgraph.laplacian(graph)
            a = i_n + laplacian
            i_d = np.identity(vector_size, dtype=np.float32)
            self.a_kron = np.kron(a.astype(np.float32), i_d.astype(np.float32))
            self.a_kron_exp = fractional_matrix_power(self.a_kron, -1 / 2).astype(np.float32)
            self.m_inverse = np.identity(self.num_users * vector_size, dtype=np.float32)  # inverse of identity is inverse

    def __init__(self, graph, num_users, cluster_data, vector_size=25, alpha=0.1):
        cluster_to_idx, idx_to_cluster = cluster_data
        cluster_to_matrix = {}
        for cluster in cluster_to_idx.Keys():
            users = cluster_to_idx[cluster]
            cluster_to_matrix[cluster] = self.ClusterToMatrix(vector_size, users, graph)
        self.context_ids_to_phis = {}  # this gets reset with every iteration, but it's good to initialize everything
        # in the __init__

    def calculate_score(self, phi, timestep, w_t):
        """
        Scores a modified long vector phi using w_t * phi, which encodes our projection of how much payoff the vector
        will produce, and the ucb, which encodes our confidence that we will gain that payoff and the potential of
        higher payoffs through further exploration
        """
        pass

    def choose(self, user_id, contexts, timestep):
        """
        Chooses best context for user, taking into account exploration, at current timestep.
        """
        pass

    def update(self, payoff, context, user_id):
        """
        Updates matrices based on payoff of chosen context
        """