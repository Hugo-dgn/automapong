import torch.nn as nn


class DQN_1(nn.Module):
    
    def __init__(self, n_inputs):
        nn.Module.__init__(self) # Tell torch that this class is a neural network

        #### Write your code here for task 12
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 64), 
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        ####
    
    def forward(self, x):
        #### Write your code here for task 12
        return self.network(x)
        ####

class DQN_2(nn.Module):
    
    def __init__(self, n_inputs):
        nn.Module.__init__(self) # Tell torch that this class is a neural network

        #### Write your code here for task 12
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 64), 
            nn.ReLU(),
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        ####
    
    def forward(self, x):
        #### Write your code here for task 12
        return self.network(x)
        ####

class DQN_3(nn.Module):
    
    def __init__(self, n_inputs):
        nn.Module.__init__(self) # Tell torch that this class is a neural network

        #### Write your code here for task 12
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 64), 
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        ####
    
    def forward(self, x):
        #### Write your code here for task 12
        return self.network(x)
        ####