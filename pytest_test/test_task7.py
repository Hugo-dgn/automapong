import numpy as np

from agents import QLearningAgent

def custom_transform_state(state):
    return (state[0],)

def test_task7():
    agent = QLearningAgent("noname", lr=0.1, gamma=0.9, eps=0.1, eeps=0, d=0.1, edecay=0.0001)
    agent.transform_state = custom_transform_state

    target = 0

    for i in range(10):

        state = (i, 0, (0, 0), (0, 0))
        action = i%3 -1
        reward = 1
        next_state = (i-1, 0, (0, 0), (0, 0))
        
        agent.learn(state, action, reward, next_state, done=False)
        
        target = agent.lr*(1 + agent.gamma * target)
        answer = agent.Q[custom_transform_state(state)][action]
        if not answer == target:
            message = f"Incorrect implementation of the Bellman equation."
            raise AssertionError(message)
    

    state = (10, 0, (0, 0), (0, 0))
    action = 0
    reward = 1
    next_state = (0, 0, (0, 0), (0, 0))

    agent.learn(state, action, reward, next_state, done=True)
    target = reward
    answer = agent.Q[custom_transform_state(state)][action]
    if not answer == target:
            message = f"The q-value of terminal states should be set to the reward itself."
            raise AssertionError(message)