import numpy as np

import pong
import agents

import train

_Env = pong.Env

def custom_render(env):
    dt, run = schedule.rander_call(env)
    return dt, run

def get_message(right_call):
    if right_call == "init":
        message = "You should instantiate the `pong.Env` class at the beginning with the following line: `env = pong.Env()`."
    elif right_call == "reset":
        message = "At the start of each episode, it's necessary to call `env.reset()`."
    elif right_call == "render":
        message = "The `pong.render` function should be called at the start of each iteration."
    elif right_call == "act":
        message = "The `agent.act` method should be called between `pong.render` and `env.step` (once for each agent)."
    elif right_call == "learn":
        message = "The `agent.learn` method should be called just after `env.step` (once for each agent)."
    elif right_call == "step":
        message = "The `env.step` method should be called after obtaining the actions chosen by the two agents."
    
    return message

class CustomEnv(pong.Env):

    def __init__(self):
        schedule.init_call()
        _Env.__init__(self)
        self.is_custom = True
    
    def step(self, action1, action2, dt=0.1):
        info = _Env.step(self, action1, action2, dt)
        schedule.step_call(action1, action2, dt, info)
        return info
    
    def reset(self):
        states = _Env.reset(self)
        schedule.reset_call(states)
        return states

class Schedule:

    def __init__(self):
        self.step = 0

        self.calls = -1
        self.last_call = None

        self.action1 = None
        self.action2 = None

        self.state1 = None
        self.state2 = None

        self.next_state1 = None
        self.next_state2 = None
    
    def get_next_call(self):
        if self.last_call is None:
            call = "init"
        
        elif self.calls == 0 and self.last_call != "reset":
            call = "reset"
        
        elif self.calls % 5 in (0, 1):
            call = "act"

        elif self.calls % 5 == 2:
            call = "step"
        
        elif self.calls % 5 in (3, 4):
            call = "learn"

        if call != "reset":
            self.calls += 1
        
        self.last_call = call
        return call

    def init_call(self):
        right_call = self.get_next_call()
        if right_call != "init":
            message = get_message(right_call)
            raise AssertionError(message)
    
    def reset_call(self, states):
        right_call = self.get_next_call()
        if right_call != "reset":
            message = get_message(right_call)
            raise AssertionError(message)
        
        self.state1, self.state2 = states
        self.next_state1, self.next_state2 = states
    
    def rander_call(self, env):
        if not isinstance(env, _Env):
            message = "pong.render takes a pon.Env object as its parameter"
            raise AssertionError(message)
        
        right_call = self.get_next_call()
        if right_call != "render":
            message = get_message(right_call)
            raise AssertionError(message)
        
        if self.step < 1000:
            return self.dt, True
        else:
            return self.dt, False
    
    def act_call(self, action, agent):
        right_call = self.get_next_call()
        if right_call != "act":
            message = get_message(right_call)
            raise AssertionError(message)

        if agent == 1:
            self.action1 = action
        else:
            self.action2 = action
    
    def learn_call(self):
        right_call = self.get_next_call()
        if right_call != "learn":
            message = get_message(right_call)
            raise AssertionError(message)

    def step_call(self, action1, action2, dt, info):
        right_call = self.get_next_call()
        if right_call != "step":
            message = get_message(right_call)
            raise AssertionError(message)

        if action1 != self.action1:
            message = "The action provided for agent1 is not correct"
            raise AssertionError(message)
        if action2 != self.action2:
            message = "The action provided for agent2 is not correct"
            raise AssertionError(message)
        
        self.state1 = self.next_state1
        self.state2 = self.next_state2

        next_state1, next_state2, reward1, reward2, done = info

        self.next_state1, self.next_state2 = next_state1, next_state2

        if done == True:
            self.calls = -2

        self.step += 1

class DummieAgent(agents.BaseAgent):

    def __init__(self, name):
        agents.BaseAgent.__init__(self, name)
    
    def act(self, state, training):
        if not training:
            message = "The `act` method's training parameters should be set to True."
            raise AssertionError(message)
        
        right_state = schedule.next_state1 if int(self.name) == 1 else schedule.next_state2

        if not (state == right_state):
            message = f"The state given to agent{int(self.name)}.act is not the right one."
            raise AssertionError(message)

        action = np.random.randint(-1, 2)
        schedule.act_call(action, int(self.name))
        return action
    
    def learn(self, state, action, reward, next_state, done):
        right_state = schedule.state1 if int(self.name) == 1 else schedule.state2
        right_next_state = schedule.next_state1 if int(self.name) == 1 else schedule.next_state2

        if not (state == right_state):
            message = f"The state given to agent{int(self.name)}.learn is not the right one."
            raise AssertionError(message)
        if not (next_state == right_next_state):
            message = f"The next_state given to agent{int(self.name)}.learn is not the right one."
            raise AssertionError(message)
        schedule.learn_call()

schedule = Schedule()

def test_task3():

    _Env = pong.Env
    _render = pong.render

    pong.Env = CustomEnv
    pong.render = custom_render

    train.train(DummieAgent("1"), DummieAgent("2"), 1000, 0.1)

    pong.Env = _Env
    pong.render = _render