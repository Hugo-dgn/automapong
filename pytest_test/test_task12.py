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
            if not isinstance(output, torch.Tensor):
                message = f"DQN_1 is not well defined. It is most likely due to a wrong implementation of the forward method. When feeding {input} to the network got {type(output)} instead of torch.Tensor"
            elif not output.shape ==torch.Size([1, 3]):
                message = f"DQN_1 is not well defined. It is most likely due to a wrong implementation of the output shape (n_input parameter). When feeding {input} to the network got {output.shape} instead of (1, 3)"
        except RuntimeError as e:
            message = f"DQN_1 is not well defined. It is most likely due to a wrong implementation of the inputs shape (n_input parameter). When feeding {input} to the network got :\n{e}"
        if not message is None:
            raise AssertionError(message)