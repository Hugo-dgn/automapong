import numpy as np

from agents import QLearningAgent
import pong

def test_task4():
    agent = QLearningAgent("noname", lr=0.1, gamma=0.9, eps=0.1, end_eps=0, d_step=0.1, eps_decay=0.0001)

    for _ in range(10):

        p = np.random.randn()
        op = np.random.randn()
        b = tuple(np.random.randn(2))
        vb = tuple(np.random.randn(2))

        state = (p, op, b, vb)

        transformed_state = agent.transform_state(state)

        if not isinstance(transformed_state, tuple):
            message = f"transform_state must return a tuple, got {transformed_state}."
            raise AssertionError(message)
        if not len(transformed_state) > 0:
            message = f"transform_state must return a non empty tuple, got {transformed_state}."
            raise AssertionError(message)
        for observation in transformed_state:
            if not isinstance(observation, (float, int)):
                message = f"transform_state must return a floats/int tuple, got {transformed_state} with an element of type {type(observation)}."
                raise AssertionError(message)