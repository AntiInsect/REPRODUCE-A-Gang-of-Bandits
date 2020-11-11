import numpy as np
from numpy.linalg import multi_dot

import scipy.sparse as sp_sparse
from scipy.linalg import fractional_matrix_power

from agents.abstract import AbstractAgent


class GOBLinAgent(AbstractAgent):
    '''
    Implementation of GOBLin algorithm
    '''

    def __init__(self, graph, num_users, dim_feature=25, alpha=0.1):
        # d in the paper
        self.dim_feature = dim_feature
        # n in the paper
        self.num_users = num_users
        # alpha is learning rate for the bais vector
        self.alpha = alpha
        self.b = np.zeros(dim_feature*num_users, dtype=np.float32)
        # the correlation matrix
        self.M = np.identity(dim_feature*num_users, dtype=np.float32)
        self.Minv = np.identity(dim_feature*num_users, dtype=np.float32)
        # the graph Laplacian and kron products
        L = sp_sparse.csgraph.laplacian(graph)
        I_n = np.identity(num_users, dtype=np.float32)
        I_d = np.identity(dim_feature, dtype=np.float32)
        self.a_kron = np.kron(L + I_n, I_d)
        self.a_kron_exp = fractional_matrix_power(self.a_kron, -1/2)
            
        self.reset()

    # this gets reset with every iteration, but it's good to initialize everything
    def reset(self):
        self.context_ids_to_phis = {} 


    def choose(self, user, contexts, timestep):
        w_t = self.Minv.dot(self.b)
        # new_contexts will contain the modified long phi vectors as described in the paper
        new_contexts = []
        for context in contexts:
            _, context_vector = context

            new_context = np.zeros(self.num_users*self.dim_feature, dtype=np.float32)
            for i in range(self.dim_feature):
                new_context[i+user*self.dim_feature] = np.float32(context_vector[i])

            # modify long vector by graph information (self.a_kron_exp) to 
            # get encoding that takes into account graph
            new_contexts.append(self.a_kron_exp @ new_context)

        scores = [w_t.dot(phi) + self.alpha * np.sqrt(multi_dot([phi.T, self.Minv, phi]) \
                                            * np.log(timestep+1)) for context in new_contexts]

        # cache long phi vectors from choose to update to avoid recomputation
        self.reset()
        for context, phi in zip(contexts, new_contexts):
            context_id, _ = context
            self.context_ids_to_phis[context_id] = phi

        return contexts[np.argmax(scores)]


    def update(self, user, context, reward):
        context_id, _ = context
        phi = self.context_ids_to_phis[context_id]
        self.b += np.real(reward * phi)
        # in order to dot properly, we need this represented as
        # a dim_feature*1 matrix, not a dim_feature-long vector
        
        phi = np.expand_dims(phi, axis=0)
        self.M += np.real(np.outer(phi, phi))
        # calculates matrix inverse (faster) using
        # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        self.Minv -= np.real(multi_dot([self.Minv, phi.T, phi, self.Minv]) / \
                            (1 + multi_dot([phi, self.Minv, phi.T]).item()))
