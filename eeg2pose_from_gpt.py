#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Define your EEGDataset class
class EEGDataset(Dataset):
    def __init__(self, eeg_data, dof_data):
        self.eeg_data = eeg_data
        self.dof_data = dof_data

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg_sample = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        dof_sample = torch.tensor(self.dof_data[idx], dtype=torch.float32)
        return eeg_sample, dof_sample

# Define your LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        output = self.fc(lstm_out)
        return output

# Set your input size, hidden size, and output size
input_size = 16  # Number of EEG channels
hidden_size = 64  # You can adjust this based on your needs
output_size = 20  # Number of body points

# Create an instance of the LSTM model
lstm_model = LSTMModel(input_size, hidden_size, output_size)

# Set your EEG data and degree of freedom (dof) data
# Replace this with your actual data loading and preprocessing
eeg_data = np.random.rand(100, 10, input_size)  # Example: 100 samples, each with a sequence length of 10
dof_data = np.random.rand(100, output_size)  # Example: Corresponding degree of freedom data

# print(dof_data)
# Split the data into training and testing sets
eeg_train, eeg_test, dof_train, dof_test = train_test_split(eeg_data, dof_data, test_size=0.2, random_state=42)

# Create instances of EEGDataset for training and testing
train_dataset = EEGDataset(eeg_train, dof_train)
test_dataset = EEGDataset(eeg_test, dof_test)

# Create DataLoader instances for training and testing
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Mean Squared Error (MSE) loss and the Adam optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
total_acc = .0
total_loss = .0
size = len(train_loader.dataset) 
for epoch in range(num_epochs):
    lstm_model.train()

    for eeg_batch, dof_batch in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = lstm_model(eeg_batch)

        # Compute the loss
        loss = criterion(outputs, dof_batch)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        total_acc += (outputs.argmax(1) == dof_batch.argmax(1)).type(torch.float).sum().item()
        total_loss += loss.item()

    average_train_loss = total_loss / len(train_loader)
    average_train_acc = total_acc / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_train_loss:.4f} - Accuracy: {average_train_acc:.4f}")

# Test the model on the test set
lstm_model.eval()
test_loss = 0.0

with torch.no_grad():
    for eeg_batch, dof_batch in test_loader:
        outputs = lstm_model(eeg_batch)
        test_loss += criterion(outputs, dof_batch).item()

average_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {average_test_loss}')
