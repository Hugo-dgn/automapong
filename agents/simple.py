import numpy as np

from agents.super import BaseAgent

class SimpleAgent(BaseAgent):
    def __init__(self, name):
        BaseAgent.__init__(self, name)
    
    def transform_state(self, state):
        (p, op, b, vb) = state
        return (p, b[1])

    def act(self, state, training):
        p, b_y = self.transform_state(state)
        return np.sign(p-b_y)