import pygame

from agents.super import BaseAgent

def _player_1():
    keys = pygame.key.get_pressed()
    if  keys[pygame.K_a]:
        return 1
    elif keys[pygame.K_q]:
        return -1
    return 0

def _player_2():
    keys = pygame.key.get_pressed()
    if  keys[pygame.K_p]:
        return 1
    elif keys[pygame.K_m]:
        return -1
    return 0

class HumanAgent(BaseAgent):
    def __init__(self, player):
        BaseAgent.__init__(self, "human")
        if player == 1:
            self.control = _player_1
        if player == 2:
            self.control = _player_2
        
        self.state_function = lambda game : [0]
        self.discretized_step = 1
        self.config = None
    
    def act(self, state, training):
        return self.control()
    
    def save(self):
        pass