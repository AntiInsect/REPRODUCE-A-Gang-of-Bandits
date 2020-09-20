import numpy
from random import *

from utils.AbstractUserContextManager import AbstractUserContextManager


class DummyUserContextManager(AbstractUserContextManager):
    def __init__(self):
        pass

    def get_user_and_contexts(self):
        contexts = []
        for i in range(5):
            contexts.append(numpy.arange(25))
        return 1, contexts

    def get_payoff(self, user_id, chosen_context):
        return randint(0, 1)
