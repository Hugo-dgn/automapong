import numpy as np
import yaml

from .physics import _move_player, _move_ball, _resolve_collision


#load configuration file
with open("pong/config.yaml", 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

class Game:

    def __init__(self, dt=0.1):
        self.p1 = 0.5 #position of player 1
        self.p2 = 0.5 #position of player 2

        self.v_p1 = 0 #speed of player 1
        self.v_p2 = 0 #speed of player 2

        #position of the ball
        self.b = np.array([0.5*config["lenght_win"], 0.5])
        
        #speed of the ball
        self.vb = np.array([1, 0])
        self.vb = self.vb/np.linalg.norm(self.vb)*config["v_b_std"]

        #count how many times players hit the ball.
        self.count_pong = 0

        self.continue_game = True
        self.draw = False

    def compute(self, dt=0.1):
        """
        compute the next frame
        """
        
        #dt is the time step for Euler integration

        self.p1 = _move_player(self.p1, self.v_p1, dt)
        self.p2 = _move_player(self.p2, self.v_p2, dt)

        self.b = _move_ball(self.b, self.vb, dt)

        if self.touche():
            self.count_pong += 1

        self.continue_game, self.vb = _resolve_collision(self.p1, self.p2, self.b, self.vb)

        if self.count_pong > 50:
            self.continue_game = False
            self.draw = True
    
    def control(self, player, action):
        """
        action = 0 : doea nothing
        action = 1 : up
        action = -1 : down
        """
        if player == 1:
            self.v_p1 = -action*config["v_p_std"]
        elif player == 2:
            self.v_p2 = -action*config["v_p_std"]
    
    def get_player_length(self):
        return config["player_lenght"]
    
    def reset(self):
        self.__init__()
    
    def win(self):
        """
        1 if player 1 wins
        2 if player 2 wins
        else 0
        """
        if not self.draw:
            if self.b[0] == config["lenght_win"] and not self.continue_game:
                return 1
            if self.b[0] == 0 and not self.continue_game:
                return 2
        return 0
    
    def touche(self):
        """
        1 if player 1 is touching the ball
        2 if player 2 is touching the ball
        else 0
        """
        if self.b[0] == config["lenght_win"] and self.continue_game:
            return 2
        if self.b[0] == 0 and self.continue_game:
            return 1
        return 0