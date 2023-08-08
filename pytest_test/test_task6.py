import numpy as np

from agents import QLearningAgent

def custom_transform_state(state):
    return (state[0],)

def test_task6():
    agent = QLearningAgent("noname", lr=0.1, gamma=0.9, eps=0.1, end_eps=0, d_step=0.1, eps_decay=0.0001)
    
    agent.transform_state = custom_transform_state

    for i in range(3):

        state = (i, 0, (0, 0), (0, 0))
        agent.Q[(i,)] = {-1 : 0, 0 : 0, 1: 0}

        agent.Q[(i,)][i-1] = 1

        action = agent.act(state, training=False)

        if not action == i-1:
            message = f"Wrong action for Q[state]={agent.Q[(i,)]}, got {action} and expected {i-1}"
            raise AssertionError(message)