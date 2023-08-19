import numpy as np

import agents

def test_task1():
    simple = agents.SimpleAgent("noname")
    for _ in range(10):
        p = np.random.randn()
        op = np.random.randn()
        b = tuple(np.random.randn(2))
        vb = tuple(np.random.randn(2))

        state = (p, op, b, vb)
        target = (p, b[1])
        answer = simple.transform_state(state)
        if not target == answer:
            if not isinstance(answer, tuple):
                message = f"transform_state does not return a tuple. Instead got {answer}"
            elif np.array(answer).shape != (2,):
                message = f"transform_state returns a tuple with wrong shape. Got {answer} with shape {np.array(answer).shape}"
            else:
                message = f"transform_state returns wrong values. Got {answer} instead of {target} with state = {(p, op, b, vb)}"
            raise AssertionError(message)