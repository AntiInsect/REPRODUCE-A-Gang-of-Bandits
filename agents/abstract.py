import abc


class AbstractAgent(abc.ABC):
    '''
    AbstractAgent operates across multiple users.
    '''

    @abc.abstractmethod
    def choose(self, user, contexts, timestep):
        pass

    @abc.abstractmethod
    def update(self, user, context, reward):
        pass
