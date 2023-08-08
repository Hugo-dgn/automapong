import network.topology as _topology

def get_topology(n):
    if hasattr(_topology, f"DQN_{n}"):
        return getattr(_topology, f"DQN_{n}")
    else:
        message = f"Topology DQN_{n} is not defined but was requested."
        raise AssertionError(message)