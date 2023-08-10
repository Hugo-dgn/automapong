import torch
import torch.nn as nn

# Set a seed for the random number generator
seed_value = 42

def set_seed():
    torch.manual_seed(seed_value)
    # If using CUDA (GPU), also set the seed for GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


class DQN_1(nn.Module):
    
    def __init__(self, n_inputs):
        set_seed()

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
    
class DQN_2(nn.Module):
    
    def __init__(self, n_inputs):
        set_seed()

        nn.Module.__init__(self) # Tell torch that this class is a neural network

        #### Write your code here for task 12
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        ####
    
    def forward(self, x):
        #### Write your code here for task 12
        return self.network(x)
        ####