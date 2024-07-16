
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim=10, hidden_dim=30, out_dim=2):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim,hidden_dim)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim,out_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x