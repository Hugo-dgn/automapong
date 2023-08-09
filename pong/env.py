#### learning Parameters ####

touch_reward = 1 #reward given each time the player touches the ball
op_touch_reward = -1
win_reward = 10 #reward given when the player won
loss_reward = -10 #reward given when the player lost
skip = 1 # agent makes a decision each skip step

#### learning Parameters ####



import numpy as np
import random
import yaml

from .game import Game

with open("pong/config.yaml", 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

#define matrices to inverse movement along an axis
inverse_y = np.array([[1, 0],
                      [0, -1]])

inverse_x = np.array([[-1, 0],
                      [0, 1]])

#flip_x - position: provides the axial symmetry
flip_x = np.array([config["lenght_win"], 0])

normalize_b = np.array([1/config["lenght_win"] , 1])


def random_start_matrix(direction):
    """
    Provides a random rotation matrix. 
    Enables a random starting point.
    """
    if direction == 0:
        direction = random.choice([-1, 1])
    theta = random.uniform(-np.pi/12, np.pi/12)

    matrix = direction*np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    return matrix

class Env:

    def __init__(self):
        self.game = Game()

        self.win_history = []

        self.skip = skip

    def get_state(self):

        normalized_vb = self.game.vb / config["v_b_std"]
        normalized_b = normalize_b * self.game.b

        state1 = (self.game.p1, self.game.p2, tuple(normalized_b), tuple(normalized_vb))
        state2 = (self.game.p2, self.game.p1, tuple(flip_x-inverse_y@normalized_b), tuple(inverse_x@normalized_vb))

        return state1, state2
    
    def get_reward(self):
        reward1 = 0
        reward2 = 0

        win = self.game.win()
        touche = self.game.touche()

        if win == 1:
            reward1 += win_reward
            reward2 += loss_reward
        elif win == 2:
            reward1 += loss_reward
            reward2 += win_reward
        
        if touche == 1:
            reward1 += touch_reward
            reward2 += op_touch_reward
        if touche == 2:
            reward2 += touch_reward
            reward1 += op_touch_reward

        return reward1, reward2

    def step(self, action1, action2, dt=0.01):
        reward1 = 0
        reward2 = 0

        for _ in range(int(self.skip)):
            self.game.control(1, action1)
            self.game.control(2, action2)

            self.game.compute(dt)

            state1, state2 = self.get_state()

            _reward1, _reward2 = self.get_reward()
            reward1 += _reward1
            reward2 += _reward2

            done = not self.game.continue_game

            if done:
                break

        return state1, state2, reward1, reward2, done

    def reset(self):
        win = self.game.win()
        if win == 0:
            direction = 0
        elif win == 1:
            direction = 1
            self.win_history.append(1)
        elif win == 2:
            direction = -1
            self.win_history.append(2)

        self.game.reset()
        random_matrix = random_start_matrix(direction)
        self.game.vb = random_matrix @ self.game.vb
        return self.get_state()

    def get_results(self):
        board = {1 : sum([1 for x in self.win_history if x==1]),
                2 : sum([1 for x in self.win_history if x==2]),
                0 : sum([1 for x in self.win_history if x==0])}

        return board