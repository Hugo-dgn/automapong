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
        pass
        ####
    
    def transform_state(self, state):
        p, op, b, vb = state
        #### Write your code here for task  4
        pass
        ####

        #### Write your code here for task 9
        pass
        ####

    def learn(self, state, action, reward, next_state, done):
        self.step += 1

        state = self.transform_state(state)
        next_state = self.transform_state(next_state)

        self.check_q_value(state)
        self.check_q_value(next_state)

        self.push(reward, done)

        #### Write your code here for task 7
        pass
        ####

    def act(self, state, training):

        state = self.transform_state(state) # transform the state in something usable
        self.check_q_value(state) # if the state is not in self.Q, add it.

        #### Write your code here for task 10
        pass
        ####

        #### Write your code here for task 11
        pass
        ####
        
        #### Write your code here for task 6
        pass
        ####
    
    def check_q_value(self, state):
        #### Write your code here for task 5
        pass
        ####