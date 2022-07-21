import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.append('models')
sys.path.append('data_gen')
from dataset import PointCloudDataset, my_collate
from models import PointNetSegmenter

#TODO: Make custom collate function for batches with point clouds of different sizes https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
num_classes = 3

# Load single mini-batch
data_dir = 'data/'
train_data = PointCloudDataset(data_dir, 'train')
# train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=my_collate)
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
inputs, labels = next(iter(train_loader))
labels = nn.functional.one_hot(labels, num_classes=num_classes).float()

# Initialize model
segmenter = PointNetSegmenter(num_classes)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(segmenter.parameters(), lr=0.003)

# Move things to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inputs, labels = inputs.to(device), labels.to(device)
# inputs = [input.to(device) for input in inputs]
# labels = [label.to(device) for label in labels]
segmenter.to(device)
# print('inputs:\n', inputs.size())

# Overtrain
losses = []
for epoch in tqdm(range(1000)):
    optimizer.zero_grad()
    logits = segmenter(inputs)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# # Predict on overtrained meshes
# segmenter.mode = 'inference' # Switch mode
# logits = segmenter(inputs)
# probs = torch.nn.functional.softmax(logits, dim=2)
# # print('probs:\n', probs)
# # print('labels\n:', labels)
# # print('Difference', criterion(logits, labels).item())

# Plot losses
plt.plot(losses)
plt.grid(True)
plt.show()
