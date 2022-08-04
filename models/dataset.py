import os
import numpy as np
import torch
from natsort import natsorted
from torch import from_numpy, int64
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self, data_folder, split_type):
        self.data_folder = data_folder
        self.split_type = split_type.lower()
        all_clouds = os.listdir(data_folder)
        self.sorted_clouds = natsorted(all_clouds)
        assert self.split_type in {'train', 'test', 'val'}

    def __getitem__(self, idx):
        cloud_loc = os.path.join(self.data_folder, self.sorted_clouds[idx])
        cloud = np.load(cloud_loc)
        cloud = from_numpy(cloud)
        pts = cloud[0:3, :].float()
        labels = cloud[-1, :].type(int64)

        return pts, labels

    def __len__(self):
        return len(self.sorted_clouds)

def my_collate(batch):
    data = [item[0].float() for item in batch]
    target = [item[1].type(int64) for item in batch]
    target = torch.LongTensor(target)

    return [data, target]

if __name__ == '__main__':
    ds = PointCloudDataset('data/', 'train')
    pts, labels = ds.__getitem__(0)
    print('pts:', pts)
    print('labels:', labels)
