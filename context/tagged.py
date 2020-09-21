import random
import numpy as np

from context.abstract import *


class Tagged(Abstract):
    '''
    For a social network with num_users users associated truly with contexts true_associations. 
    For sample_context, returns a random collection of context vectors such that one is
    truly associated with the user. To compute reward, returns 1 if the context is truly associated
    with the user and zero otherwise.
    '''

    def __init__(self, num_users, true_associations, contexts):
        self.true_associations = true_associations
        self.contexts = contexts
        self.num_users = num_users
        
        self.context_dict = {}
        for context in self.contexts:
            self.context_dict[context[0]] = context

    def sample_user(self):
        return random.randrange(0, self.num_users)

    def sample_contexts(self, user):
        truth_context_id = random.choice(self.true_associations[user])
        contexts = random.choices(self.contexts, k=24) + [self.context_dict[truth_context_id]]
        random.shuffle(contexts)
        return contexts

    def sample_reward(self, user, contexts):
        return (contexts[0] in self.true_associations[user])