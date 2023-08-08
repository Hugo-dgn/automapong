import torch

import network

def test_task12():
    DQN = network.get_topology(1)

    for n in range(1, 11):
        dqn = DQN(n)

        input = torch.Tensor([i for i in range(n)]).unsqueeze(0)

        message = None
        try:
            output = dqn(input)
        except RuntimeError as e:
            message = f"DQN_1 is not well defined. It is most likely due to a wrong implementation of the inputs shape (n_input parameter). When feeding {input} to the network got :\n{e}"
        if not message is None:
            raise AssertionError(message)