from AbstractAgent import AbstractAgent
import numpy as np
import scipy.sparse as sp_sparse
from scipy.linalg import fractional_matrix_power
from numpy.linalg import multi_dot
import math


class BlockAgent(AbstractAgent):
    """
    Implementation of GOBLin Block algorithm
    """
    def __init__(self, graph, num_users, cluster_data, vector_size=25, alpha=0.1):
        pass
