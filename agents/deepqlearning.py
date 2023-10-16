import random
from collections import deque

import numpy as np
import torch

from agents.super import BaseAgent

device = "cuda" if torch.cuda.is_available() else "cpu"

class DeepQLearningAgent(BaseAgent):
    def __init__(self, name, dqn, lr, gamma, eps, eeps, edecay, capacity, batch, tau, skip):
        BaseAgent.__init__(self, name)
        self.lr = lr #learning rate
        self.memory = ReplayMemory(capacity)
        self.batch = batch #batch size
        self.tau = tau #soft update parameter
        self.skip = skip #learning step

        self.gamma = gamma #Bellman equation
        self.eps = eps #start value of epsilon for epsilon greedy algorithm
        self.eeps = eeps #end value of epsilon for epsilon greedy algorithm
        self.edecay = edecay #speed of the transition between eps and eeps

        self.step = 0 #number of call of learn

        dummie_state = (0, 0, (0, 0), (0, 0))

        n = len(self.transform_state(dummie_state))

        self.dqn = dqn(n) #policy net
        self._target_dqn = dqn(n) #target net

        self.dqn.to(device)
        self._target_dqn.to(device)

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr, amsgrad=True) #optimizer
        self.criterion = torch.nn.SmoothL1Loss() #loss function
    
    def init(self):
        self.dqn.to(device)
        self._target_dqn.to(device)
    
    def transform_state(self, state):
        #### Write your code here for task 13
        pass
        ####

    def learn(self, state, action, reward, next_state, done):
        self.step += 1

        state = self.transform_state(state)
        next_state = self.transform_state(next_state)

        self.push(reward, done)

        #### Write your code here for task 17
        pass
        ####

    def act(self, state, training):

        state = self.transform_state(state) # transform the state in something usable
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        #epsilon greedy
        if training and np.random.random() < (self.eps - self.eeps) * np.exp(-self.edecay * self.step) + self.eeps:
            return np.random.choice([-1, 0, 1])
        
        #### Write your code here for task 14
        pass
        ####
    
    def soft_update_target(self):
        #### Write your code here for task 16
        pass
        ####
    
    def get_loss(self):
        transition = self.memory.sample(self.batch)

        state = transition["state"]
        action = transition["action"] + 1
        next_state = transition["next_state"]
        reward = transition["reward"]
        done = transition["done"]
        
        #### Write your code here for task 15
        pass
        ####

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
    
def _get_transition(states, actions, next_states, rewards, dones):
    transition = {
        "state" : torch.tensor(states, dtype=torch.float32, device=device),
        "action" : torch.tensor(actions, dtype=torch.int64, device=device),
        "next_state" : torch.tensor(next_states, dtype=torch.float32, device=device),
        "reward" : torch.tensor(rewards, device=device),
        "done" : torch.tensor(dones, device=device)
    }
    return transition