from DummyAgent import DummyAgent
from LinUCBAgent import LinUCBAgent

def load_agent(algorithm_name, num_features, alpha):
    if (algorithm_name == "dummy"):
        return DummyAgent()
    elif (algorithm_name == "linucb"):
        return LinUCBAgent(num_features, alpha)
    elif (algorithm_name == "goblin"):
        #return GOBLinAgent()
        pass
    else:
        print("Algorithm not implemented")