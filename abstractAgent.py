import abc

class AbstractAgent(abc.ABC):
  
  @abc.abstractmethod
  def get_payoff(self):
    pass

  def update(self):
    pass
