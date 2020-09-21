import random

from agents.abstract import AbstractAgent

class DummyAgent(AbstractAgent):
    def __init__(self):
        pass

    def choose(self, user, contexts, steps):
        a = random.randint(0, len(contexts) - 1)
        return contexts[a]

    def update(self, user, context, reward):
        pass
