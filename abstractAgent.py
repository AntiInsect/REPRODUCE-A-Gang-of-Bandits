import abc

class abstractAgent(abc.ABC):
  
  @abc.abstractmethod
  def get_payoff(self):
    pass

  def update(self):
    pass
