import numpy as np

from agents.super import BaseAgent

class SimpleAgent(BaseAgent):
    def __init__(self, name):
        BaseAgent.__init__(self, name) # Cela permet de sauvegarder l'agent et ses performances
    
    def transform_state(self, state):
        p, op, b, vb = state
        ####write yout code here for task 1
        return (p, b[1])
        ####

    def act(self, state, training):
        state = self.transform_state(state)
        ####write yout code here for task 2
        p, b_y = state
        return np.sign(p-b_y)
        ####