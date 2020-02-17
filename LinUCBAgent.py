from AbstractAgent import AbstractAgent
import numpy as np
import random as rd
import math
from numpy.linalg import multi_dot
from collections import defaultdict


class LinUCBAgent(AbstractAgent):
    """
    Implementation of LinUCB algorithm
    """

    class MatrixBias:
        """
        Maintains a matrix and bias for each user in the algorithm
        Both the matrix and bias represent information learned by chosen contexts and rewards
        """

        def __init__(self, num_features):
            self.num_features = num_features
            self.M = np.identity(num_features, dtype=np.float32)
            self.b = np.zeros(num_features, dtype=np.float32)
            self.Minv = np.linalg.inv(self.M)

        def update(self, payoff, context):
            # convert context into an np array
            new_context = np.zeros(self.num_features, dtype=np.float32)
            for i in range(len(context[1])):
                new_context[i] = context[1][i]
            # update self.b
            self.b = self.b + new_context * np.float32(payoff)
            # in order to dot properly, we need this represented as a vector_size*1 matrix, not a vector_size-long vector
            new_context = np.expand_dims(new_context, axis=0)
            new_context_transpose = np.transpose(new_context)
            outer_product = np.matmul(new_context_transpose, new_context)
            self.M = self.M + outer_product
            # calculates matrix inverse using https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
            numerator = multi_dot([self.Minv, new_context_transpose, new_context, self.Minv])
            self.Minv = self.Minv - (numerator / (1 + multi_dot([new_context, self.Minv, new_context_transpose]).item()))

    def __init__(self, num_features, alpha=0.1, is_sin=False):
        # maintains user matrix and bias
        self.num_features = num_features
        self.user_information = defaultdict(lambda: self.MatrixBias(num_features))
        self.alpha = alpha
        self.is_sin = is_sin

    def choose(self, user_id, contexts, timestep):
        """
        Chooses best context for user, taking into account exploration, at current timestep.
        """
        # If LinUCB-SIN, then use only one matrix_and_bias instance -- i.e., every user is treated as user 0
        if self.is_sin:
            user_id = 0
        matrix_and_bias = self.user_information[user_id]
        b = matrix_and_bias.b
        Minv = matrix_and_bias.Minv

        # Construct matrix A inverse times b
        w = np.dot(Minv, b)
        w_t = np.transpose(w)

        # we need to obtain a score for every context
        scores = []
        for i in range(0, len(contexts)):
            # Calculate scores
            cur_con = [np.float32(contexts[i][1][j]) for j in range(len(contexts[i][1]))]
            ucb = self.alpha * np.sqrt(multi_dot([np.transpose(cur_con), Minv, cur_con])
                                                    * math.log(timestep + 1))
            scores.append(float(w_t.dot(cur_con) + ucb))
        # get the best score and return it
        best_idx = np.argmax(scores)
        return contexts[best_idx]

    def update(self, payoff, context, user_id):
        """
        Updates matrices based on payoff of chosen context
        """
        # If LinUCB-SIN, we are updating only user_id 0
        if self.is_sin:
            user_id = 0
        # Update A and b vectors
        matrix_and_bias = self.user_information[user_id]
        matrix_and_bias.update(payoff, context)
