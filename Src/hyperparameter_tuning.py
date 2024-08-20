import torch
import torch.optim as optim
from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt

def hyperparameter_tuning(X_train_tensor, y_train_tensor, input_size, hidden_size, output_size, num_hidden_layers, num_epochs=5):
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]
    activation_functions = [nn.ReLU(), nn.LeakyReLU(), nn.ELU()]
    
    results = defaultdict(list)
    
    for lr, batch_size, activation in product(learning_rates, batch_sizes, activation_functions):
        print(f"Training with LR={lr}, Batch Size={batch_size}, Activation={activation}")
        
        model = MLP(input_size, hidden_size, output_size, num_hidden_layers, activation)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        criterion = nn.MSELoss()
        
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        train_losses = []
        train_accuracies = []
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataset)
            train_losses.append(epoch_loss)
            train_accuracy = calculate_accuracy(model, dataloader)
            train_accuracies.append(train_accuracy)
        
        results['learning_rate'].append(lr)
        results['batch_size'].append(batch_size)
        results['activation_function'].append(activation)
        results['train_losses'].append(train_losses)
        results['train_accuracies'].append(train_accuracies)
    
    return results

def plot_hyperparameter_tuning_results(results, num_epochs):
    plt.figure(figsize=(15, 10))
    for i in range(len(results['learning_rate'])):
        plt.subplot(3, 3, i + 1)
        plt.plot(range(1, num_epochs + 1), results['train_losses'][i], label='Training Loss')
        plt.plot(range(1, num_epochs + 1), results['train_accuracies'][i], label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(f'LR={results["learning_rate"][i]}, Batch Size={results["batch_size"][i]}, Activation={results["activation_function"][i]}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
