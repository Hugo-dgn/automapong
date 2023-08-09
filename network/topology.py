import torch.nn as nn


class DQN_1(nn.Module):
    
    def __init__(self, n_inputs):
        nn.Module.__init__(self) # Tell torch that this class is a neural network

        #### Write your code here for task 12
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        ####
    
    def forward(self, x):
        #### Write your code here for task 12
        return self.network(x)
        ####