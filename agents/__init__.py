from agents.human import HumanAgent
from agents.simple import SimpleAgent
from agents.super import BaseAgent
from agents.qlearning import QLearningAgent
from agents.deepqlearning import DeepQLearningAgent
from agents.random import RandomAgent
from agents.strongagent import StrongAgent

import torch as _torch

def load(name):
    path = "agents/save/"
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    with open(path+name, 'rb') as f:
        agent = _torch.load(f, map_location=device)
    agent.init()
    return agent

def get_agent(name_agent):
    if name_agent == "human1":
        return HumanAgent(player=1)
    elif name_agent == "human2":
        return HumanAgent(player=2)
    elif name_agent == "simple":
        agent = SimpleAgent("simple")
        def _save():
            pass
        agent.save = _save
        return agent
    elif name_agent == "strong":
        agent = StrongAgent("strong")
        def _save():
            pass
        agent.save = _save
        return agent
    else:
        return load(name_agent)