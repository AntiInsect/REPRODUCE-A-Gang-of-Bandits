import abc

class AbstractUserContextManager(abc.ABC):

  @abc.abstractmethod
  def get_user_and_contexts(self):
    pass

  @abc.abstractmethod
  def get_payoff(self):
    pass
