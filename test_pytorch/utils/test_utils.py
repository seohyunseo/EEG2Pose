import torch

def test_model(model, test_dataloader):
    model.eval()
    mse_loss = torch.nn.MSELoss()
    size = len(test_dataloader.dataset)

    total_loss = .0
    total_acc = .0

    with torch.no_grad():
        for features, labels in test_dataloader:
            preds = model(features)
            loss = mse_loss(preds, labels)
            total_loss += loss.item()
            
            # Calculate accuracy-like metric
            total_acc += (preds.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()
    
    average_test_loss = total_loss / len(test_dataloader)
    average_test_acc = total_acc / size
    print(f"Test Loss: {average_test_loss:.4f} - Test Accuracy: {average_test_acc:.4f}")
