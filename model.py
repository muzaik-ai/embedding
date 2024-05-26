import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, middle_layer_count=5):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.middle_layers = nn.ModuleList([
            nn.Linear(hidden_size * (i + 1), hidden_size * (i + 2)) for i in range(middle_layer_count)
        ])
        self.fc3 = nn.Linear(hidden_size * (middle_layer_count + 1), output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for layer in self.middle_layers:
            x = F.relu(layer(x))
        x = self.fc3(x)
        return x
    
    def complexity(self):
        total_complexity = self.fc1.in_features * self.fc1.out_features
        for layer in self.middle_layers:
            total_complexity += layer.in_features * layer.out_features
        total_complexity += self.fc3.in_features * self.fc3.out_features
        return total_complexity