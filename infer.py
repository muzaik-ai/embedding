import torch
import numpy as np
from model import Model
from huggingface_hub import hf_hub_download
import features as ft

models = [
    'embedding_model_v1_nano.pth',
    'embedding_model_v1_small.pth',
    'embedding_model_v1_medium.pth',
    'embedding_model_v1_large.pth',
    'embedding_model_v1_xlarge.pth',
]

model_name = 'embedding_model_v1_nano.pth'

checkpoint = torch.load(hf_hub_download('muzaik/embedding_model_v1', filename=model_name))


input_size = int(checkpoint['input_size'])
hidden_size = checkpoint['hidden_size']
output_size = checkpoint['output_size']
middle_layer_count = checkpoint['middle_layer_count']
complexity = checkpoint['complexity']
model_state_dict = checkpoint['model_state_dict']

print(f"Complexity of Model: {complexity/1e3:.0f}K")
model = Model(input_size=input_size, hidden_size=hidden_size, output_size=output_size, middle_layer_count=middle_layer_count)
model.load_state_dict(model_state_dict)
model.eval()

def infer(file_path):
    if isinstance(file_path, str):
        features = np.load(file_path)
    else:
        features = file_path
    features = ft.average_features(features)
    features = features.reshape(-1)  # Flatten the features
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(features)
    return embedding.squeeze().numpy()

if __name__ == "__main__":
    audio_file = 'test.mp3'
    sample = ft.extract_features(audio_file)
    embedding = infer(sample)
    print(f"Embedding for {audio_file}: {embedding}")
    print(f"Embedding shape: {embedding.shape}")
    print(f'Embedding size: {embedding.size}')