import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.basic_model import BasicNet
from dataset.dataset import CustomDataset

def train_model(model, train_dataloader, num_epochs=100):
    mse_loss = torch.nn.MSELoss()
    adam_optimizer = optim.Adam(model.parameters())
    size = len(train_dataloader.dataset)

    for epoch in range(num_epochs):
        model.train()

        total_loss = .0
        total_acc = .0

        for features, labels in train_dataloader:
            
            adam_optimizer.zero_grad()
            preds = model(features)
            loss = mse_loss(preds, labels)
            loss.backward()
            adam_optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy-like metric
            total_acc += (preds.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()

        average_train_loss = total_loss / len(train_dataloader)
        average_train_acc = total_acc / size

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {average_train_loss:.4f} - Accuracy: {average_train_acc:.4f}")