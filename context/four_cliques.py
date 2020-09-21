import uuid

import random
import numpy as np
from sklearn.preprocessing import normalize

from context.abstract import *


class FourCliques(Abstract):
    '''
    For a social network with 4 cliques, each with 25 users, assigns each user within a 
    clique a random vector of size 25 representing the "ideal" context vector for that user.
    Returns random context vectors for any user when getting users and contexts. 
    Computes reward as user_vector dot context_vector + uniform distribution within epsilon --
    Epsilon is reward noise.
    '''

    CLIQUE_SIZE = 25
    NUM_CLIQUES = 4
    PROVIDED_CONTEXTS = 10

    def __init__(self, epsilon=0.0, dim_feature=25):
        self.user_vectors = []
        # user_vectors will contain NUM_CLIQUES * CLIQUE_SIZE vectors, CLIQUE_SIZE of the same vector for each clique
        self.epsilon = epsilon
        self.dim_feature = dim_feature

        for _ in range(self.NUM_CLIQUES):
            rand_vector = self.generate_rand_vector()
            for _ in range(self.CLIQUE_SIZE):
                self.user_vectors.append(rand_vector)
        
    def sample_user(self):
        return random.randrange(0, self.NUM_CLIQUES * self.CLIQUE_SIZE)

    def sample_contexts(self, user):
        # since 4cliques has no "real" contexts, we generate PROVIDED_CONTEXTS context vectors on the fly
        # to be chosen from for our chosen user
        # contexts are associated with a unique identifier, in 4cliques, as each context is uniquely generated,
        # we generate a unique identifier for each context before releasing it. For other datasets, this unique
        # identifier is provided in the dataset.
        return [(uuid.uuid1(), self.generate_rand_vector()) for i in range(self.PROVIDED_CONTEXTS)]

    def sample_reward(self, user, contexts):
        # reward is dotted user_vector and context_vector plus a random sample bounded by epsilon
        return np.dot(self.user_vectors[user], contexts[1]) + \
                np.random.uniform(-self.epsilon, self.epsilon)

    def generate_rand_vector(self):
        rand_vector = np.random.uniform(low=-1, high=1, size=(self.dim_feature,))
        rand_vector = normalize(rand_vector[:,np.newaxis], axis=0).ravel()
        return rand_vector

    @classmethod
    def generate_cliques(cls, threshold):
        graph = np.zeros((100, 100))
        # creates a block adjacency matrix with 4 25 x 25 blocks of ones
        # along the diagonal corresponding to each clique
        for i in range(cls.NUM_CLIQUES):
            for j in range(cls.CLIQUE_SIZE):
                for k in range(cls.CLIQUE_SIZE):
                    graph[j + i * cls.CLIQUE_SIZE][k + i * cls.CLIQUE_SIZE] = 1

        noise_generated = np.random.rand(cls.NUM_CLIQUES * cls.CLIQUE_SIZE, cls.NUM_CLIQUES * cls.CLIQUE_SIZE)
        # get top triangle of matrix without diagonal, and create symmetrical matrix
        noise_top = np.triu(noise_generated, 1)
        noise = noise_top + np.transpose(noise_top)

        def check_threshold(element):
            return (element > threshold)

        # vectorizing a function makes it apply elementwise to a matrix
        vectorized_above_threshold = np.vectorize(check_threshold)
        above_threshold = vectorized_above_threshold(noise)
        # swap values where the noise is above the threshold
        result = np.logical_xor(graph, above_threshold)
        # logical xor returns trues and falses, we need ones and zeroes, which we produce with another
        # vectorized function
        convert_from_true_false_to_1_0 = np.vectorize(lambda x: 1 if x else 0)
        return convert_from_true_false_to_1_0(result)
