import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.append('models')
sys.path.append('data_gen')
from dataset import PointCloudDataset
from models import PointNetSegmenter
from unit_shape_gen import set_axes_equal

data_dir = 'data/'
train_data = PointCloudDataset(data_dir, 'train')
train_loader = DataLoader(train_data, batch_size=3, shuffle=True)
# data_iter = iter(train_loader)
inputs, labels = next(iter(train_loader))
inputs, labels = inputs.float(), labels.type(torch.LongTensor)
transformed_labels = labels.view(-1, 1)[:, 0]

num_classes = 2
# Initialize model
segmenter = PointNetSegmenter(num_classes)
criterion = nn.NLLLoss()
optimizer = optim.Adam(segmenter.parameters(), lr=0.001)

# Move things to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inputs = inputs.to(device)
transformed_labels = transformed_labels.to(device)
segmenter.to(device)

# Overtrain
losses = []
for epoch in tqdm(range(50)):
    optimizer.zero_grad()
    output = segmenter(inputs)
    output = output.view(-1, num_classes)
    loss = criterion(output, transformed_labels)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Predict on overtrained meshes
output = segmenter(inputs)
one_hot_lables = nn.functional.one_hot(labels, num_classes)
print('output shape:', output.shape)
print('labels shape:', one_hot_lables.shape)
print('final output:', segmenter(inputs).cpu() - one_hot_lables)
plt.plot(losses)
plt.show()
