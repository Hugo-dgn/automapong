import numpy as np

from agents.super import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, name, alpha, gamma, eps, end_eps, d_step, eps_decay):
        BaseAgent.__init__(self, name)
        self.Q = {}

        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.end_eps = end_eps
        self.eps_decay = eps_decay

        self.d_step = d_step

        self.step = 0
    
    def discretize(self, s):
        #### Write your code here for task 9
        d_s = tuple([int(x/self.d_step)*self.d_step for x in s])

        return d_s
        ####
    
    def transform_state(self, state):
        #### Write your code here for task  4
        p, op, b, vb = state
        s = (np.clip(p-b[1], -0.2, 0.2), b[0]/40, vb[0]/40, np.sign(vb[1]))
        ####

        #### Write your code here for task 9
        d_s = self.discretize(s)
        ####

        return d_s

    def learn(self, state, action, reward, next_state, done):
        state = self.transform_state(state)
        next_state = self.transform_state(next_state)

        self.check_q_value(state)
        self.check_q_value(next_state)

        self.push(reward, done)

        #### Write your code here for task 7

        max_next_q = max(self.Q[next_state].values())
        if done:
            self.Q[state][action] = reward
        else:
            self.Q[state][action] += self.alpha * (reward + self.gamma * max_next_q - self.Q[state][action])
        
        ####

    def act(self, state, training):
        self.step += 1

        state = self.transform_state(state) # transform the state in something usable
        self.check_q_value(state) # if the state is not in self.Q, add it.

        #### Write your code here for task 9

        if training and np.random.random() < (self.eps - self.end_eps) * np.exp(-self.eps_decay * self.step) + self.end_eps:
            return np.random.choice([-1, 0, 1])
        
        ####
        
        #### Write your code here for task 6

        action_q_values = self.Q[state]
        if all(value == list(action_q_values.values())[0] for value in action_q_values.values()):
            return np.random.choice([-1, 0, 1])

        return max(action_q_values, key=action_q_values.get)
    
        ####
    
    def check_q_value(self, state):
        #### Write your code here for task 5
        if state not in self.Q:
            self.Q[state] = {1 : 0, -1 : 0, 0 : 0}
        ####