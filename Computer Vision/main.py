
import torch
import numpy as np
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.datasets import load_wine


class WineDataset(Dataset):
    def __init__(self):
        data = load_wine()
        X, y = data['data'], data['target']
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.n_samples = self.y.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples


##########
dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)
# num_workers = might help accelarate loading

total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)

for epoch in range(2):
    for i, (inputs, labels) in enumerate(dataloader):
        print(inputs)
        break
