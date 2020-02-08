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

    class ClusterInfo:
        """
        Maintains necessary vectors/matrices for each cluster:
        bias, m, m_inverse, a_kron_exp, num_users
        """
        def __init__(self, vector_size, users, graph):
            self.num_users = len(users)
            self.user_to_user_in_cluster = {}
            # create mapping between user_ids and user index in matrix
            for i in range(self.num_users):
                self.user_to_user_in_cluster[users[i]] = i
            # create graph for this cluster -- new adjacency matrix
            new_graph = np.zeros((self.num_users, self.num_users))
            for first_user in users:
                for second_user in users:
                    if graph[first_user][second_user] == 1:
                        new_graph[self.user_to_user_in_cluster[first_user]] \
                                 [self.user_to_user_in_cluster[second_user]] = 1
                    else:
                        new_graph[self.user_to_user_in_cluster[first_user]] \
                                 [self.user_to_user_in_cluster[second_user]] = 0
            # initiate necessary vectors/matrices for this cluster
            self.bias = np.zeros(vector_size * self.num_users, dtype=np.float32)
            self.m = np.identity(self.num_users * vector_size, dtype=np.float32)
            i_n = np.identity(self.num_users, dtype=np.float32)
            # construct a laplacian matrix based on the graph, that we will modify and then
            # take the kronecker product of to get a representation of the graph that helps us learn
            # although the laplacian is sp_sparse, if called on a dense matrix it will return a dense matrix
            laplacian = sp_sparse.csgraph.laplacian(new_graph)
            a = i_n + laplacian
            i_d = np.identity(vector_size, dtype=np.float32)
            self.a_kron = np.kron(a.astype(np.float32), i_d.astype(np.float32))
            self.a_kron_exp = fractional_matrix_power(self.a_kron, -1 / 2).astype(np.float32)
            self.m_inverse = np.identity(self.num_users * vector_size, dtype=np.float32)  # inverse of identity is inverse

    def __init__(self, graph, num_users, cluster_data, vector_size=25, alpha=0.1):
        self.vector_size = vector_size
        self.alpha = alpha
        self.cluster_to_idx, self.idx_to_cluster = cluster_data
        self.cluster_info = {}
        for cluster in self.cluster_to_idx.keys():
            users = self.cluster_to_idx[cluster]
            self.cluster_info[cluster] = self.ClusterInfo(vector_size, users, graph)
        self.context_ids_to_phis = {}  # this gets reset with every iteration, but it's good to initialize everything
        # in the __init__

    def calculate_score(self, phi, timestep, w_t, cluster):
        """
        Scores a modified long vector phi using w_t * phi, which encodes our projection of how much payoff the vector
        will produce, and the ucb, which encodes our confidence that we will gain that payoff and the potential of
        higher payoffs through further exploration
        """
        m_inverse = self.cluster_info[cluster].m_inverse
        ucb = self.alpha * np.sqrt(multi_dot([np.transpose(phi), m_inverse, phi]) * math.log(timestep + 1))
        return float(w_t.dot(phi) + ucb)

    def choose(self, user_id, contexts, timestep):
        """
        Chooses best context for user, taking into account exploration, at current timestep.
        """
        cluster = self.idx_to_cluster[user_id]
        cluster_info = self.cluster_info[cluster]
        user_id = cluster_info.user_to_user_in_cluster[user_id]
        w_t = cluster_info.m_inverse.dot(cluster_info.bias)
        # new_contexts will contain the modified long phi vectors as described in the paper
        new_contexts = []
        for context in contexts:
            context_id, context_vector = context
            new_context = np.zeros(cluster_info.num_users * self.vector_size, dtype=np.float32)
            for i in range(self.vector_size):
                # the context_vector is placed in the new context such that it begins in block indexed by the
                # current user, this provides a means of identifying to the algorithm which user is currently being
                # examined. The block begins at user_id * self.vector_size, following which the vector is placed, with
                # a length vector_size
                new_context[i + user_id * self.vector_size] = np.float32(context_vector[i])
            # modify long vector by graph information (self.a_kron_exp) to get encoding that takes into account graph
            new_contexts.append(np.matmul(cluster_info.a_kron_exp, new_context))
        scores = [self.calculate_score(context, timestep, w_t, cluster) for context in new_contexts]
        max_context_index = np.argmax(scores)
        # cache long phi vectors from choose to update to avoid recomputation
        self.context_ids_to_phis = {}
        for context, phi in zip(contexts, new_contexts):
            context_id, context_vector = context
            self.context_ids_to_phis[context_id] = phi
        return contexts[max_context_index]

    def update(self, payoff, context, user_id):
        """
        Updates matrices based on payoff of chosen context
        """
        cluster = self.idx_to_cluster[user_id]
        cluster_info = self.cluster_info[cluster]
        context_id, context_vector = context
        # retrieve modified long vector phi associated with the context_id and stored in self.choose
        phi = self.context_ids_to_phis[context_id]
        cluster_info.bias = cluster_info.bias + phi * payoff
        # in order to dot properly, we need this represented as a vector_size*1 matrix, not a vector_size-long vector
        phi = np.expand_dims(phi, axis=0)
        phi_transpose = np.transpose(phi)
        outer_product = np.matmul(phi_transpose, phi)
        cluster_info.m = cluster_info.m + outer_product
        # calculates matrix inverse using https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        numerator = multi_dot([cluster_info.m_inverse, phi_transpose, phi, cluster_info.m_inverse])
        cluster_info.m_inverse = cluster_info.m_inverse - (numerator / (1 + multi_dot([phi, cluster_info.m_inverse, phi_transpose]).item()))
