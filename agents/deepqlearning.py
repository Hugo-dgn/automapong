import random
import contextlib

from collections import deque
import numpy as np

import torch

from utils import get_agent

from agents.super import BaseAgent

from play import benchmark

device = "cuda" if torch.cuda.is_available() else "cpu"

class DeepQLearningAgent(BaseAgent):
    def __init__(self, name, dqn, lr, gamma, eps, eeps, edecay, capacity, batch, tau, skip):
        BaseAgent.__init__(self, name)
        self.lr = lr
        self.memory = ReplayMemory(capacity)
        self.batch = batch
        self.tau = tau
        self.skip = skip

        self.gamma = gamma
        self.eps = eps
        self.eeps = eeps
        self.edecay = edecay

        self.step = 0

        dummie_state = (0, 0, (0, 0), (0, 0))

        n = len(self.transform_state(dummie_state).squeeze(0))

        self.dqn = dqn(n)
        self._target_dqn = dqn(n)

        self.dqn.to(device)
        self._target_dqn.to(device)

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = torch.nn.SmoothL1Loss()
    
    def transform_state(self, state):
        #### Write your code here for task
        p, op, b, vb = state
        s = (p, b[0], b[1], vb[0], vb[1])
        ####

        return torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

    def learn(self, state, action, reward, next_state, done):
        self.step += 1

        state = self.transform_state(state)
        next_state = self.transform_state(next_state)

        self.push(reward, done)

        #### Write your code here for task
        self.memory.push(state, action, next_state, reward, done)

        if len(self.memory) >= self.batch and self.step % self.skip == 0:
            self.soft_update_target()
            loss = self.get_loss()

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(self.dqn.parameters(), 100)
            self.optimizer.step()

        ####

    def act(self, state, training):

        state = self.transform_state(state) # transform the state in something usable

        if training and np.random.random() < (self.eps - self.eeps) * np.exp(-self.edecay * self.step) + self.eeps:
            return np.random.choice([-1, 0, 1])
        
        #### Write your code here for task

        with torch.no_grad():
            q_values = self.dqn(state).squeeze(0)

        action = torch.argmax(q_values).item() - 1

        return action
    
        ####
    
    def soft_update_target(self):
        target_net_state_dict = self._target_dqn.state_dict()
        dqn_net_state_dict = self.dqn.state_dict()

        for key in dqn_net_state_dict:
            target_net_state_dict[key] = dqn_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        
        self._target_dqn.load_state_dict(target_net_state_dict)
    
    def get_loss(self):
        #### Write your code here for task
        transition = self.memory.sample(self.batch)
        
        state = transition["state"]
        action = transition["action"] + 1
        next_state = transition["next_state"]
        reward = transition["reward"]
        done = transition["done"]

        out = self.dqn(state)
        qvlues = out.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            expected_qvalues = self.gamma * self._target_dqn(next_state).max(1).values * (1 - done) + reward

        loss = self.criterion(qvlues, expected_qvalues)

        return loss

        ####

def _get_transition(states, actions, next_states, rewards, dones):
    transition = {
        "state" : torch.cat(states),
        "action" : torch.tensor(actions, dtype=torch.int64, device=device),
        "next_state" : torch.cat(next_states),
        "reward" : torch.tensor(rewards, device=device),
        "done" : torch.tensor(dones, device=device)
    }
    return transition

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
    
    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, int(done)))

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)

        return _get_transition(*zip(*sample))

    def __len__(self):
        return len(self.memory)

    def __reduce__(self):
        return (ReplayMemory, (self.capacity,))