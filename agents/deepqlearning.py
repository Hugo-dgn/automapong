import random

from collections import deque
import numpy as np

import torch

from agents.super import BaseAgent

device = "cuda" if torch.cuda.is_available() else "cpu"

class DeepQLearningAgent(BaseAgent):
    def __init__(self, name, DQN, lr, gamma, eps, end_eps, eps_decay, capacity, batch, skip, update):
        BaseAgent.__init__(self, name)
        self.lr = lr
        self.memory = ReplayMemory(capacity)
        self.batch = batch
        self.skip = skip
        self.update = update

        self.gamma = gamma
        self.eps = eps
        self.end_eps = end_eps
        self.eps_decay = eps_decay

        self.step = 0

        dummie_state = (0, 0, (0, 0), (0, 0))

        n = len(self.transform_state(dummie_state).squeeze(0))

        self.dqn = DQN(n)
        self.target_dqn = DQN(n)

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = torch.nn.SmoothL1Loss()
    
    def transform_state(self, state):
        #### Write your code here for task  4
        p, op, b, vb = state
        s = (p, b[0], b[1], vb[0])
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
            self.update_target()
            loss = self.get_loss()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        ####

    def act(self, state, training):

        state = self.transform_state(state) # transform the state in something usable

        if training and np.random.random() < (self.eps - self.end_eps) * np.exp(-self.eps_decay * self.step) + self.end_eps:
            return np.random.choice([-1, 0, 1])
        
        #### Write your code here for task

        with torch.no_grad():
            q_values = self.dqn(state).squeeze(0)

        action = torch.argmax(q_values).item() - 1

        print(action)

        return action
    
        ####
    
    def update_target(self):
        if self.step % self.update == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
    
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
            expected_qvalues = self.gamma * self.target_dqn(next_state).max(1).values * (1 - done) + reward
        
        loss = self.criterion(qvlues, expected_qvalues)

        return loss

        ####
    
    def init(self):
        #### Write your code here for task
        pass
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