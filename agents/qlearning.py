import numpy as np

from agents.super import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, name, lr, gamma, eps, eeps, d, edecay):
        BaseAgent.__init__(self, name)
        self.Q = {}

        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eeps = eeps
        self.edecay = edecay

        self.d = d

        self.step = 0
    
    def discretize(self, s):
        #### Write your code here for task 9
        return tuple(map(lambda x : int(x/self.d)*self.d, s))
        ####
    
    def transform_state(self, state):
        p, op, b, vb = state
        #### Write your code here for task  4
        state = (p, b[1])
        ####

        #### Write your code here for task 9
        return self.discretize(state)
        ####

    def learn(self, state, action, reward, next_state, done):
        self.step += 1

        state = self.transform_state(state)
        next_state = self.transform_state(next_state)

        self.check_q_value(state)
        self.check_q_value(next_state)

        self.push(reward, done)

        #### Write your code here for task 7
        if done:
            self.Q[state][action] = reward
        else:
            self.Q[state][action] += self.lr * (reward + self.gamma * max(self.Q[next_state].values()) - self.Q[state][action])
        ####

    def act(self, state, training):

        state = self.transform_state(state) # transform the state in something usable
        self.check_q_value(state) # if the state is not in self.Q, add it.

        #### Write your code here for task 10
        if training and np.random.random() < (self.eps - self.eeps) * np.exp(-self.edecay * self.step) + self.eeps:
            return np.random.choice((-1, 0, 1))
        ####

        #### Write your code here for task 11
        if self.Q[state][-1] == self.Q[state][0] == self.Q[state][1]:
            return np.random.choice((-1, 0, 1))
        ####
        
        #### Write your code here for task 6
        action = max(self.Q[state], key=self.Q[state].get)
        return action
        ####
    
    def check_q_value(self, state):
        #### Write your code here for task 5
        if not state in self.Q:
            self.Q[state] = {-1 : 0, 0 : 0, 1 : 0}
        ####