from collections import defaultdict

import numpy as np
from numpy.linalg import multi_dot

from agents.abstract import AbstractAgent


class LinUCBAgent(AbstractAgent):
    '''
    Implementation of LinUCB algorithm
    '''

    class CorrelationBias:
        '''
        Maintains a correlation matrix and bias vector
        for each user which both represent information
        learned by chosen arms' context and sampled reward
        '''

        def __init__(self, dim_feature):
            self.dim_feature = dim_feature
            # the bias vector
            self.b = np.zeros(self.dim_feature, dtype=np.float32)
            # the correlation matrix and its inversion
            # no need to perform inversion on an identity matrix
            self.M = np.identity(self.dim_feature, dtype=np.float32)
            self.Minv = np.identity(self.dim_feature, dtype=np.float32)

        def update(self, context, reward):
            # convert context into an np array
            new_context = np.array(context[1], dtype=np.float32)
            # update self.b
            self.b += np.float32(reward) * new_context

            # in order to dot properly, we need this represented as 
            # a vector_size*1 matrix, not a vector_size-long vector
            new_context = np.array(np.expand_dims(new_context, axis=0))
            outer_product = np.outer(new_context, new_context)
            self.M += outer_product
            # calculates matrix inverse (faster) using 
            # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
            self.Minv -= multi_dot([self.Minv, new_context.T, new_context, self.Minv]) / \
                        (1 + multi_dot([new_context, self.Minv, new_context.T]).item())

    def __init__(self, dim_feature, alpha=0.1, is_sin=False):
        self.is_sin = is_sin
        self.dim_feature = dim_feature
        # A defaultdict will never raise a KeyError
        # Any key that does not exist gets the value returned by the default factory
        self.user_infor = defaultdict(lambda: self.CorrelationBias(dim_feature))
        # the learning rate for the bais vector
        self.alpha = alpha

    def choose(self, user, contexts, timestep):
        '''
        Chooses best arm's context for user taking into account exploration at current timestamp
        '''

        # If LinUCB-SIN, then use only one correlation_bias instance
        # i.e., every user is treated as user 0
        if self.is_sin: user = 0

        # the estimation of the user preference
        b = self.user_infor[user].b
        Minv = self.user_infor[user].Minv
        w_t = np.transpose(Minv.dot(b))

        # Calculate scores for every arm's context
        scores = []
        for i in range(len(contexts)):
            cur_con = np.float32(contexts[i][1])
            ucb = w_t.dot(cur_con) + \
                    self.alpha * np.sqrt(multi_dot([cur_con.T, Minv, cur_con]) * np.log(timestep+1))
            scores.append(ucb)
        # get the best score idx and return it corresponding centext
        return contexts[np.argmax(scores)]

    def update(self, user, context, reward):
        '''
        Updates matrices based on reward of chosen context
        '''

        # If LinUCB-SIN, we are updating only user 0
        if self.is_sin: user = 0
            
        # Update A and b vectors
        self.user_infor[user].update(context, reward)
