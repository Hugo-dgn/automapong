import numpy as np

from agents import QLearningAgent

def test_task4():
    agent = QLearningAgent("noname", lr=0.1, gamma=0.9, eps=0.1, end_eps=0, d_step=0.1, eps_decay=0.0001)

    if not len(agent.Q) == 0:
        message = "The Q table must be initialized empty."
        raise AssertionError(message)
    
    for i in range(10):
        state = (i, i+1)
        agent.check_q_value(state)

        if not state in agent.Q:
            message = "New states are not added to the Q table."
            raise AssertionError(message)

        target = {-1 : 0, 0 : 0, 1 : 0}
        answer = agent.Q[state]
        if not answer == target:
            message = f"New states in the Q table are not set with the dictionary. {target}."
            raise AssertionError(message)
    
        agent.Q[state][0] = 100
        agent.check_q_value(state)

        new_target = {-1 : 0, 0 : 100, 1 : 0}
        new_answer = agent.Q[state]
        if not new_answer == new_target:
            message = "Already existent states are being overwritten."
            raise AssertionError(message)
    
    if not len(agent.Q) == 10:
        message = "Some states are being removed from the Q table."
        raise AssertionError(message)