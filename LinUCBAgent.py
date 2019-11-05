from AbstractAgent import AbstractAgent
import random
import numpy as np
import random as rd

class LinUCBAgent(AbstractAgent):

    class MatrixBias:
        def __init__(self, int: num_features):
            self.M = np.identity(num_features)
            self.b = np.zeros(num_features)

        def update(self, payoff, context):
            self.M += np.dot(context, np.transpose(context))
            self.b += np.dot(context, payoff)

    def __init__(self, int: num_features, int: alpha, int: num_actions):
        # maintains user matrix and bias
        self.user_information = defaultdict(lambda: MatrixBias(num_features))
        self.d = num_features
        self.alpha = alpha
        self.K = num_actions

    def choose(self, user_id, contexts):
        # Obtain user matrix and bias
        matrix_and_bias = user_information[user_id]
        M = matrix_and_bias.M
        b = matrix_and_bias.b

        # Construct matrix A inverse times b
        Minv =  np.linalg.inv(M)
        w = np.dot(Minv, b)

        # we need to obtain a UCB values for every action
        best_a = -1
        ucb = -np.inf
        for a in range(0, self.K):
            # Calculate UCB
            cur_con = contexts[a]
            cur_con_T = np.transpose(cur_con)
            cur_ucb = np.dot(np.transpose(w), cur_con) + \
                      alpha * np.sqrt(np.transpose(np.dot(np.dot(cur_con_T, Minv), cur_con))
            # retain best action, ties broken randomly
            if cur_ucb > ucb:
                best_a, ucb = a, cur_ucb
            elif cur_ucb == ucb:
                best_a = rd.choice([a, best_a])

        return best_a, contexts[best_a]

    def update(self, payoff, context, user_id):
        # Update A and b vectors
        matrix_and_bias = self.user_information[user_id]
        matrix_and_bias.update(payoff, context)