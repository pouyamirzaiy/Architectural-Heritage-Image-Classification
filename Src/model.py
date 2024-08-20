import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(activation)
        self.layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(activation)
            self.layers.append(nn.Dropout(0.5))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))
        self.layers.append(nn.Sigmoid())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
