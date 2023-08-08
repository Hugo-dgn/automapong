import numpy as np

import agents

def test_task2():
    simple = agents.SimpleAgent("noname")
    for _ in range(10):
        p = np.random.randn()
        op = np.random.randn()
        b = np.random.randn(2)
        vb = np.random.randn(2)

        state = (p, op, b, vb)
        target = np.sign(p-b[1])
        answer = simple.act(state, training=False)
        if not target == answer:
            if not isinstance(answer, (float, int)):
                message = f"the action of an agent must be a number. Instead got {answer}"
            if not answer in [-1, 0, 1]:
                message = f"the action of an agent must be 1, -1 or 0. Instead got {answer}"
            else:
                message = f"agent simple does not return the right action : expected {target} ; given {answer} for p = {p} and b[1] = {b[1]}"
            
            raise AssertionError(message)