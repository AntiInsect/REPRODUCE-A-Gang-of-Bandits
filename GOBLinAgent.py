from AbstractAgent import AbstractAgent
import numpy
import scipy.sparse as scp_sp
from scipy.linalg import fractional_matrix_power
from numpy.linalg import multi_dot
import math

class GOBLinAgent(AbstractAgent):
    def __init__(self, graph, num_users, vector_size=25, alpha=2):
        self.vector_size = vector_size
        self.num_users = num_users
        self.alpha = alpha
        self.bias = numpy.zeros(vector_size * num_users)
        self.m = numpy.identity(num_users * vector_size)
        i_n = numpy.identity(num_users)
        laplacian = scp_sp.csgraph.laplacian(graph)
        a = i_n + laplacian
        i_d = numpy.identity(vector_size)
        self.a_kron = numpy.kron(a, i_d)
        self.a_kron_exp = fractional_matrix_power(self.a_kron, -1/2)

    def calculate_score(self, phi, timestep, w_t):
        ucb = self.alpha * numpy.sqrt(multi_dot([numpy.transpose(phi), numpy.linalg.inv(self.m), phi]) * math.log(timestep + 1))
        return float(numpy.dot(w_t, phi) + ucb)
    
    def choose(self, user_id, contexts, timestep):
        m_inverse = numpy.linalg.inv(self.m)
        w_t = numpy.matmul(m_inverse, self.bias)
        new_contexts = []
        for context in contexts:
            context_id, context_vector = context
            new_context = numpy.zeros(self.num_users * self.vector_size)
            for i in range(self.vector_size):
                new_context[i + user_id] = context_vector[i]
            new_contexts.append(numpy.matmul(self.a_kron_exp, new_context))
        scores = [self.calculate_score(context, timestep, w_t) for context in new_contexts]
        max_context_index = numpy.argmax(scores)
        self.context_ids_to_phis = {}
        for context, phi in zip(contexts, new_contexts):
            context_id, context_vector = context
            self.context_ids_to_phis[context_id] = phi
        return contexts[max_context_index]
        
        
    def update(self, payoff, context, user_id):
        context_id, context_vector = context
        phi = self.context_ids_to_phis[context_id]
        self.m = self.m + numpy.matmul(phi, numpy.transpose(phi))
        self.bias = self.bias + phi * payoff
