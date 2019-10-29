from AbstractAgent import AbstractAgent
import random

class DummyAgent(AbstractAgent):
    def __init__(self):
        pass

    def choose(self, user_id, contexts):
        return random.choice(contexts)

    def update(self, payoff):
        pass