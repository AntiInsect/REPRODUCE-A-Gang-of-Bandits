from DummyAgent import DummyAgent

def load_agent(algorithm_name):
    if (algorithm_name == "dummy"):
        return DummyAgent()
    else:
        print("Algorithm not implemented")