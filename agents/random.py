import numpy as np

from agents.super import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, name):
        BaseAgent.__init__(self, name)
    
    def transform_state(self, state):
        return (0,)

    def act(self, state, training):
        return np.random.choice([-1, 0, 1])