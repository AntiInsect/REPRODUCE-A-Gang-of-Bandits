import abc

class AbstractAgent(abc.ABC):
  
  @abc.abstractmethod
  def get_payoff(self):
    pass

  @abc.abstractmethod
  def update(self):
    pass
