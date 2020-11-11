import random

from agents.abstract import AbstractAgent

class DummyAgent(AbstractAgent):
    def __init__(self):
        pass

    def choose(self, user, contexts, steps):
        return contexts[random.randint(0, len(contexts)-1)]

    def update(self, user, context, reward):
        pass
