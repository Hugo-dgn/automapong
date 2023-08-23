import torch

import agents
import network

def test_task13():
    dqn = network.get_topology(1)
    message = None
    try:
        agent = agents.DeepQLearningAgent("noname", dqn=dqn, lr=0.1, gamma=0.9, eps=0.1, eeps=0, edecay=1, capacity=1000, batch=32, tau=0.1, skip=1)
    #I want to catch the error that is raised when the transform_state function is not implemented
    except Exception as e:
        message = f"The DeepQLearningAgent couldn't be initialized. It is most likely due to a wrong implementation of `transform_state`. Got the following error:\n {e}"
    if not message is None:
        raise AssertionError(message)
    for i in range(10):
        state = (i, i, (i, i), (i, i))

        answer = agent.transform_state(state)

        if not isinstance(answer, torch.Tensor):
            message = f"For deep Q learning, `transform_state` must return a torch.Tensor instance. Got {answer} with type {type(answer)}"
            raise AssertionError(message)
        if not (answer.shape[0] == 1 and len(answer.shape) == 2):
            message = f"For deep Q learning, `transform_state` must return a torch.Tensor instance with shape (1, k). Got {answer} with shape {answer.shape}"
            raise AssertionError(message)