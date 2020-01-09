from AbstractAgent import AbstractAgent
import numpy as np
import scipy as sp
import scipy.sparse as sp_sparse
from scipy.linalg import fractional_matrix_power
from numpy.linalg import multi_dot
import math

class GOBLinAgent(AbstractAgent):
    def __init__(self, graph, num_users, vector_size=25, alpha=2):
        self.vector_size = vector_size
        self.num_users = num_users
        # alpha is measure of learning rate
        self.alpha = alpha
        self.bias = np.zeros(vector_size * num_users)
        self.m = np.identity(num_users * vector_size)
        i_n = np.identity(num_users)
        laplacian = sp_sparse.csgraph.laplacian(graph)
        a = i_n + laplacian
        i_d = np.identity(vector_size)
        self.a_kron = np.kron(a, i_d)
        self.a_kron_exp = fractional_matrix_power(self.a_kron, -1/2)
        self.m_inverse = np.identity(num_users * vector_size) # inverse of identity is inverse


    def calculate_score(self, phi, timestep, w_t):
        ucb = self.alpha * np.sqrt(multi_dot([np.transpose(phi), self.m_inverse, phi]) * math.log(timestep + 1))
        return float(w_t.dot(phi) + ucb)
    
    def choose(self, user_id, contexts, timestep):
        w_t = self.m_inverse.dot(self.bias)
        new_contexts = []
        for context in contexts:
            context_id, context_vector = context
            new_context = np.zeros(self.num_users * self.vector_size)
            for i in range(self.vector_size):
                new_context[i + user_id] = context_vector[i]
            new_contexts.append(np.matmul(self.a_kron_exp, new_context))
        scores = [self.calculate_score(context, timestep, w_t) for context in new_contexts]
        max_context_index = np.argmax(scores)
        self.context_ids_to_phis = {}
        for context, phi in zip(contexts, new_contexts):
            context_id, context_vector = context
            self.context_ids_to_phis[context_id] = phi
        return contexts[max_context_index]
        
        
    def update(self, payoff, context, user_id):
        context_id, context_vector = context
        phi = self.context_ids_to_phis[context_id]
        phi = np.expand_dims(phi, axis=0)
        self.bias = self.bias + np.squeeze(phi) * payoff
        outer_product = np.matmul(np.transpose(phi), phi)
        self.m = self.m + outer_product
        # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        phi_transpose = np.transpose(phi)
        numerator = multi_dot([self.m_inverse, outer_product, self.m_inverse])
        self.m_inverse = self.m_inverse - (numerator / (1 + multi_dot([phi, self.m_inverse, phi_transpose]).item()))
