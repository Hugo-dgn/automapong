import random
from collections import deque

import torch

from agents.deepqlearning import _get_transition

import agents
import network

class DummieReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

        self._current_sample = None
    
    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, int(done)))

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)

        self._current_sample = _get_transition(*zip(*sample))

        return self._current_sample

    def __len__(self):
        return len(self.memory)

    def __reduce__(self):
        return (DummieReplayMemory, (self.capacity,))

def compute_loss(agent):
    transition = agent.memory._current_sample
    state = transition["state"]
    action = transition["action"] + 1
    next_state = transition["next_state"]
    reward = transition["reward"]
    done = transition["done"]

    out = agent.dqn(state)
    qvlues = out.gather(1, action.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        expected_qvalues = agent.gamma * agent._target_dqn(next_state).max(1).values * (1 - done) + reward

    loss = agent.criterion(qvlues, expected_qvalues)

    return loss

def test_task15():

    dqn = network.get_topology(1)
    agent = agents.DeepQLearningAgent("noname", dqn=dqn, lr=0.1, gamma=0.9, eps=0.1, eeps=0, edecay=1, capacity=1000, batch=32, tau=0.1, skip=1)
    agent.memory = DummieReplayMemory(agent.memory.capacity)

    for i in range(100):
        state = (i, i, (i, i), (i, i))
        next_state = (i+1, i+1, (i+1, i+1), (i+1, i+1))
        action = i%3 - 1
        reward = i

        state = agent.transform_state(state)
        next_state = agent.transform_state(next_state)
        agent.memory.push(state, action, next_state, reward, i%2 == 0)
    
    for j in range(10):

        answer = agent.get_loss()
        target = compute_loss(agent)

        if not isinstance(answer, torch.Tensor):
            message = f"The loss must be a torch.Tensor instance. Got {answer}."
            raise AssertionError(message)
        elif not answer.shape == torch.Size([]):
            message = f"The shape of the loss must be {torch.Size([])} (meaning a float). Got {answer} with shape {answer.shape}."
            raise AssertionError(message)
        elif not answer == target:
            message = f"The value of the loss is wrong. Expected {target} but got {answer}."
            raise AssertionError(message)

