import abc


class AbstractUserContextManager(abc.ABC):
    '''
    Abstract class for user context manager
    '''
    @abc.abstractmethod
    def sample_user(self):
        pass

    @abc.abstractmethod
    def sample_contexts(self, user):
        pass

    @abc.abstractmethod
    def sample_reward(self, user, contexts):
        pass