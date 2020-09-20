import abc


class AbstractAgent(abc.ABC):
    '''
    AbstractAgent operates across multiple users.
    '''

    @abc.abstractmethod
    def choose(self, user_id, contexts, timestep):
        pass

    @abc.abstractmethod
    def update(self, payoff, context, user_id):
        pass
