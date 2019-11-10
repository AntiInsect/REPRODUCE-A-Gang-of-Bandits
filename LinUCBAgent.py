from AbstractAgent import AbstractAgent
import random
import numpy as np
import random as rd
<<<<<<< HEAD
from collections import defaultdict

class LinUCBAgent(AbstractAgent):

    class MatrixBias:
        def __init__(self, num_features):
=======

class LinUCBAgent(AbstractAgent):
    '''
    Implementation of LinUCB algorithm
    '''

    class MatrixBias:
        '''
        Maintains a matrix and bias for each user in the algorithm
        Both the matrix and bias represent information learned by chosen contexts and rewards
        '''
        def __init__(self, int: num_features):
>>>>>>> 0b89b368ee8ad6fb48cf46d6713530e4288d2982
            self.M = np.identity(num_features)
            self.b = np.zeros(num_features)

        def update(self, payoff, context):
            self.M += np.dot(context, np.transpose(context))
            self.b += np.dot(context, payoff)

<<<<<<< HEAD
    def __init__(self, num_features, alpha = 2):
        # maintains user matrix and bias
        self.d = num_features
        self.user_information = defaultdict(lambda: self.MatrixBias(num_features))
        self.alpha = alpha

    def choose(self, user_id, contexts, t):
        matrix_and_bias = self.user_information[user_id]
=======
    def __init__(self, int: num_features, int: num_actions, int: alpha = 2):
        # maintains user matrix and bias
        self.user_information = defaultdict(lambda: MatrixBias(num_features))
        self.d = num_features
        self.alpha = alpha
        self.K = num_actions

    def choose(self, user_id, contexts, int: t):
        # Obtain user matrix and bias
        matrix_and_bias = user_information[user_id]
>>>>>>> 0b89b368ee8ad6fb48cf46d6713530e4288d2982
        M = matrix_and_bias.M
        b = matrix_and_bias.b

        # Construct matrix A inverse times b
<<<<<<< HEAD
        Minv =  np.linalg.inv(M)
=======
        Minv = np.linalg.inv(M)
>>>>>>> 0b89b368ee8ad6fb48cf46d6713530e4288d2982
        w = np.dot(Minv, b)

        # we need to obtain a UCB values for every action
        best_a = -1
        ucb = -np.inf
<<<<<<< HEAD
        for a in range(0, len(contexts)):
=======
        for a in range(0, self.K):
>>>>>>> 0b89b368ee8ad6fb48cf46d6713530e4288d2982
            # Calculate UCB
            cur_con = contexts[a]
            cur_con_T = np.transpose(cur_con)
            cur_ucb = np.dot(np.transpose(w), cur_con) + \
<<<<<<< HEAD
                      self.alpha * np.sqrt(np.transpose(np.dot(np.dot(cur_con_T, Minv), cur_con) * np.log(t + 1)))
            # retain best action, ties broken randomly
            #print(ucb, cur_ucb)
=======
                      alpha * np.sqrt(np.transpose(np.dot(np.dot(cur_con_T, Minv), cur_con)) * np.log(t + 1))
            # retain best action, ties broken randomly
>>>>>>> 0b89b368ee8ad6fb48cf46d6713530e4288d2982
            if cur_ucb > ucb:
                best_a, ucb = a, cur_ucb
            elif cur_ucb == ucb:
                best_a = rd.choice([a, best_a])

        return best_a, contexts[best_a]

    def update(self, payoff, context, user_id):
        # Update A and b vectors
        matrix_and_bias = self.user_information[user_id]
        matrix_and_bias.update(payoff, context)