import abc

class AbstractAgent(abc.ABC):
  
  @abc.abstractmethod
  def choose(self):
    pass

  @abc.abstractmethod
  def update(self):
    pass
