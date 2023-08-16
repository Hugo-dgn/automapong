import torch

import agents
import network

def test_task14():
    dqn = network.get_topology(1)
    agent = agents.DeepQLearningAgent("noname", dqn=dqn, lr=0.1, gamma=0.9, eps=0.1, eeps=0, edecay=1, capacity=1000, batch=32, tau=0.1, skip=1)

    for i in range(10):
        state = (i, i, (i, i), (i, i))
        tstate = agent.transform_state(state)

        qvalues = agent.dqn(tstate)
        target = torch.argmax(qvalues) - 1
        answer = agent.act(state, False)

        if not target == answer:
            message = f"Wrong implementation of the act methode for deepqlearning agent. Got action {answer} instead of {target} with qvalues {qvalues}."
            raise AssertionError(message)