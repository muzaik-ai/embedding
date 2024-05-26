import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.file_names = [f for f in os.listdir(dataset_dir) if f.endswith('.npy')]
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.dataset_dir, self.file_names[idx])
        features = np.load(file_path)
        features = torch.tensor(features, dtype=torch.float32)
        features = features.view(-1)
        return features