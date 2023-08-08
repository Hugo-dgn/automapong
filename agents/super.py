import copy

import torch
import numpy as np

def smooth_history(history):
    if len(history) > 1_000_000:
        n = int(len(history)/100)
        window = np.ones(n) / n
        smooth_history = np.convolve(np.array(history), window, mode='valid').tolist()
        smooth_history = smooth_history[::int(len(history)/100)]
        return smooth_history
    else:
        return history

class BaseAgent:

    def __init__(self, name):

        self.name = name

        self._reward_history = []
        self.train_games = 0
        self._current_game_reward = []
    
    def push(self, reward, done):
        self._current_game_reward.append(reward)
        if done:
            self.train_games += 1
            self._reward_history.append(np.mean(self._current_game_reward))
            self._current_game_reward = []
    
    def learn(self, state, action, reward, next_state, done):
        pass
    
    def before_save(self):
        pass

    def save(self):
        self._reward_history = smooth_history(self._reward_history)

        path = "agents/save/"
        with open(path+self.name, 'wb') as f:
            torch.save(self, f)
    
    def init(self):
        pass