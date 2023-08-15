import numpy as np
import utils

from agents.super import BaseAgent

config = utils.get_config()

class StrongAgent(BaseAgent):
    def __init__(self, name):
        BaseAgent.__init__(self, name)
    
    def transform_state(self, state):
        (p, op, b, vb) = state
        return (p, b[0], b[1], vb[0], vb[1])

    def act(self, state, training):
        p, b_x, b_y, vb_x, vb_y = self.transform_state(state)

        if vb_x > 0:
            return np.sign(p-1/2)

        tan_alpha = -vb_y / vb_x

        target = b_y + config["lenght_win"] * b_x * tan_alpha

        while True:
            if target > 1:
                target = 2 - target
            elif target < 0:
                target = -target
            else:
                break

        return np.sign(p-target)