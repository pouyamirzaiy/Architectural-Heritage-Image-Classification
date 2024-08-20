import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing import preprocess_data
from eda import eda
from model import MLP
from train import train_model, calculate_accuracy
from hyperparameter_tuning import hyperparameter_tuning, plot_hyperparameter_tuning_results
from evaluation import evaluate_model
from visualization import plot_training_results, plot_feature_importance

# Load and preprocess data
file_path = 'path/to/your/dataset.csv'
df = preprocess_data(file_path)

# Perform EDA
eda(df)

# Split data into train and test sets
X = df.drop('No-show', axis=1)
y = df['No-show']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

# Define model parameters
input_size = X.shape[1]
hidden_size = 128
output_size = 1
num_hidden_layers = 2
num_epochs = 30
batch_size = 64

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, criterion, and optimizer
model = MLP(input_size, hidden_size, output_size, num_hidden_layers)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Train the model
model = train_model(model, criterion, optimizer, train_loader, num_epochs)

# Evaluate the model
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
mse, mae, accuracy, precision, recall, f1 = evaluate_model(model, test_loader)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Perform hyperparameter tuning
results = hyperparameter_tuning(X_train_tensor, y_train_tensor, input_size, hidden_size, output_size, num_hidden_layers)
plot_hyperparameter_tuning_results(results, num_epochs)

# Plot training results
plot_training_results(results['train_losses'][0], results['train_accuracies'][0], num_epochs)

# Plot feature importance
plot_feature_importance(df)
