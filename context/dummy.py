import random
import numpy as np

from context.abstract import *


class DummyUserContextManager(AbstractUserContextManager):
    '''
    Dummy user context manager for testing
    '''

    def __init__(self):
        pass

    def sample_user(self):
        return 1

    def sample_contexts(self, user):
        return [np.arange(25) for i in range(5)]

    def sample_reward(self, user, contexts):
        return random.randint(0, 1)