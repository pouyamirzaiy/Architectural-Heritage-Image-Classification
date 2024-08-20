import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, criterion, optimizer, dataloader, num_epochs=30):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    return model

def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            predicted = torch.round(outputs).squeeze()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    accuracy = correct / total
    return accuracy
