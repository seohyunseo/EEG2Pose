import torch
import torch.nn as nn

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        output, _ = self.gru(x)
        # output shape: (batch_size, sequence_length, hidden_size)
        flattened_output = self.flatten(output)
        # flattened_output shape: (batch_size, sequence_length * hidden_size)
        output = self.fc(flattened_output)
        output = self.softmax(output)
        return output
