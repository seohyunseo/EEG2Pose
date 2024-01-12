import torch
import torch.nn as nn

class BasicNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(BasicNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 26),
            nn.Linear(26, 26),
            nn.Linear(26, output_size)
        )

    def forward(self, x):
        return self.layers(x)
