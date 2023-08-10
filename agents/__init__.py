from agents.human import HumanAgent
from agents.simple import SimpleAgent
from agents.super import BaseAgent
from agents.qlearning import QLearningAgent
from agents.deepqlearning import DeepQLearningAgent
from agents.random import RandomAgent


import torch as _torch

def load(name):
    path = "agents/save/"
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    with open(path+name, 'rb') as f:
        agent = _torch.load(f, map_location=device)
    agent.init()
    return agent