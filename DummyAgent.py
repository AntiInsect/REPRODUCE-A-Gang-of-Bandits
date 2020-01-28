from AbstractAgent import AbstractAgent
import random


class DummyAgent(AbstractAgent):
    def __init__(self):
        pass

    def choose(self, user_id, contexts, steps):
        a = random.randint(0, len(contexts) - 1)
        return a, contexts[a]

    def update(self, payoff, context, user_id):
        pass
