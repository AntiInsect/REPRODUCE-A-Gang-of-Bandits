from DummyUserContextManager import DummyUserContextManager

def load_data(dataset_location):
    if (dataset_location == "dummy"):
        return DummyUserContextManager(), None
    else:
        print("Algorithm not implemented")
        return None, None