import numpy as np

from agents import QLearningAgent

def test_task9():
    agent = QLearningAgent("noname", lr=0.1, gamma=0.9, eps=0.1, end_eps=0, d_step=0.1, eps_decay=0.0001)

    for i in range(1, 11):

        d_step = i/10
        agent.d_step = d_step

        s = tuple(10*np.random.random(i))

        target = tuple([int(x/d_step)*d_step for x in s])
        answer = agent.discretize(s)

        if not target == answer:
            message = f"Wrong implementation of discretize methode. Expected {target} but got {answer} with d_step={d_step} and s={s}."
            raise AssertionError(message)