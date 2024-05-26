import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import Dataset
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters, experiment, and model configuration
version = 'v1' # -> basic linear model, v2 -> CNN model (in works lol)
size = 'xxx'
exp_name = f'embedding_model_{version}_{size}'
input_size = 128*10
hidden_size = 256
middle_layer_count = 2
output_size = 128*10
learning_rate = 0.001
batch_size = 32
epochs = 100


dataset = Dataset('dataset')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Model(input_size=input_size, hidden_size=hidden_size, output_size=output_size, middle_layer_count=middle_layer_count)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()
criterion = criterion.to(device)
model = model.to(device)

complexity = model.complexity()
# param count in thousands (K)
print(f"Param count of Model: {complexity/1e3:.0f}K")

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        target = batch.view(batch.size(0), -1)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}| Train Loss: {epoch_train_loss:.4f}")

format = {
    'input_size': input_size,
    'hidden_size': hidden_size,
    'output_size': output_size,
    'middle_layer_count': middle_layer_count,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'complexity': complexity,
}

torch.save(format, f'{exp_name}.pth')
