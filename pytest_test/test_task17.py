import torch

import agents

from train import train

from agents.deepqlearning import ReplayMemory

n = 1000

capacity=1000
batch=32
skip=1


class DummyTopology(torch.nn.Module):
    def __init__(self, n):
        torch.nn.Module.__init__(self)
        self.layer = torch.nn.Linear(1, 3)

    def forward(self, x):
        return self.layer(x)

class DummyReplyMemory(ReplayMemory):

    def __init__(self, capacity):
        ReplayMemory.__init__(self, capacity)
    
    def push(self, state, action, next_state, reward, done):
        scheduler.transition_pushed = True
        scheduler.check()
        ReplayMemory.push(self, state, action, next_state, reward, done)

class Scheduler:

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.transition_pushed = False

        self.calculated_loss = False
        self.soft_updated = False

        self.zero_grad = False

        self.backward = False

        self.optimizer_stepped = False
    
    def check(self):
        if not self.transition_pushed:
            if any((self.calculated_loss, self.soft_updated, self.zero_grad, self.backward, self.optimizer_stepped)):
                message = "You should call memory.push before calling get_loss, soft_update_target, optimizer.zero_grad, backward or optimizer.step."
                raise Exception(message)
        elif not (self.calculated_loss and self.soft_updated):
            if any((self.zero_grad, self.backward, self.optimizer_stepped)):
                message = "You should call get_loss and soft_update_target before calling optimizer.zero_grad, backward or optimizer.step."
                raise Exception(message)
        elif not self.zero_grad:
            if any((self.backward, self.optimizer_stepped)):
                message = "You should call optimizer.zero_grad before calling backward or optimizer.step."
                raise Exception(message)
        elif not self.backward:
            if self.optimizer_stepped:
                message = "You should call backward before calling optimizer.step."
                raise Exception(message)
    
    def complete_check(self, agent):
        if not (len(agent.memory) > batch and agent.step % agent.skip == 0):
            if not self.transition_pushed:
                message = "You didn't call memory.push."
                raise Exception(message)
            if any((self.calculated_loss, self.soft_updated, self.zero_grad, self.backward, self.optimizer_stepped)):
                message = "You should not call get_loss, soft_update_target, optimizer.zero_grad, backward or optimizer.step if len(memory) <= batch or step % skip != 0."
                raise Exception(message)
        elif not all((self.transition_pushed, self.calculated_loss, self.soft_updated, self.zero_grad, self.backward, self.optimizer_stepped)):
            message = "You should call memory.push, get_loss, soft_update_target, optimizer.zero_grad, backward and optimizer.step in this order.\n"

            calls = []
            if self.transition_pushed:
                calls.append("memory.push")
            if self.calculated_loss:
                calls.append("get_loss")
            if self.soft_updated:
                calls.append("soft_update_target")
            if self.zero_grad:
                calls.append("optimizer.zero_grad")
            if self.backward:
                calls.append("backward")
            if self.optimizer_stepped:
                calls.append("optimizer.step")
            
            message += "You called: " + ", ".join(calls) + "."
            raise Exception(message)

def dummy_transform_state(state):
    return torch.tensor([1], dtype=torch.float32).unsqueeze(0)

def dummy_get_loss(self):
    loss = agents.DeepQLearningAgent.get_loss(self)
    scheduler.calculated_loss = True
    scheduler.check()
    def dummy_backward(backward):
        backward()
        scheduler.backward = True
        scheduler.check()
    backward = loss.backward
    loss.backward = lambda: dummy_backward(backward)
    return loss

def dummy_soft_update_target(self):
    agents.DeepQLearningAgent.soft_update_target(self)
    scheduler.soft_updated = True
    scheduler.check()

def dummy_optimizer_zero_grad(zero_grad):
    zero_grad()
    scheduler.zero_grad = True
    scheduler.check()

def dummy_optimizer_step(step):
    step()
    scheduler.optimizer_stepped = True
    scheduler.check()

scheduler = Scheduler()

def test_task17():
    dqn = DummyTopology
    agent = agents.DeepQLearningAgent("noname", dqn=dqn, lr=0.1, gamma=0.9, eps=0.1, eeps=0, edecay=1, capacity=capacity, batch=batch, tau=0.1, skip=skip)
    agent.transform_state = dummy_transform_state
    agent.memory = DummyReplyMemory(1000)
    agent.get_loss = lambda: dummy_get_loss(agent)
    agent.soft_update_target = lambda: dummy_soft_update_target(agent)

    zero_grad = agent.optimizer.zero_grad
    agent.optimizer.zero_grad = lambda: dummy_optimizer_zero_grad(zero_grad)

    step = agent.optimizer.step
    agent.optimizer.step = lambda: dummy_optimizer_step(step)

    for i in range(n):
        state = (i, i, (i, i), (i, i))
        next_state = (i+1, i+1, (i+1, i+1), (i+1, i+1))

        action = i % 3 - 1
        reward = i % 10
        done = i % 100 == 0

        agent.learn(state, action, reward, next_state, done)
        scheduler.complete_check(agent)
        scheduler.reset()