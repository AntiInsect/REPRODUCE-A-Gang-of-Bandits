from AbstractUserContextManager import AbstractUserContextManager
from random import *
import numpy

class DummyUserContextManager(AbstractUserContextManager):
    def __init__(self):
        pass

    def get_user_and_contexts(self):
        return 1, numpy.ndarray([1, 2, 3])

    def get_payoff(self, user_id, chosen_context):
        return randint(0, 1)
