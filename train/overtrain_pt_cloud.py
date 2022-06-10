import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('models')
sys.path.append('data_gen')
from models import PointNetSegmenter
from unit_shape_gen import set_axes_equal

# Load point cloud
cloud_0 = np.load('data/sample_point_cloud.npy')
cloud_0 = torch.from_numpy(cloud_0)
xyz_data = torch.unsqueeze(cloud_0[0:3, :], 0).float()
labels = torch.unsqueeze(cloud_0[-1, :], 0)
print('xyz data type:', xyz_data.type())

# Initialize model
segmenter = PointNetSegmenter(2)
segmenter = segmenter

# Run data through model
print('input shape:', xyz_data.shape)
output = segmenter(xyz_data)
print('output shape:', output.shape)
print('ground truth shape:', labels.shape)

# # Prepare axes
# fig = plt.figure()
# ax = plt.axes(projection='3d')

# # Plot point cloud
# x, y, z, w, labels = pt_cloud_0
# ax.scatter(x, y, z, s=2, c=labels)

# # Finish formating axis
# set_axes_equal(ax)
# plt.show()
