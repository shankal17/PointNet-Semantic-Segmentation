import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformerDimK(nn.Module):
    def __init__(self, k=64):
        super(SpatialTransformerDimK, self).__init__()

        # Shared weight layers
        self.conv_1 = nn.Conv1d(k, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024, 1)

        # Fully connected layers
        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, k**2)

        # Batch norms
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

        # Other
        self.k = k

    def forward(self, x):
        batch_size = x.size()[0]
        out = F.relu(self.bn_1(self.conv_1(x)))
        out = F.relu(self.bn_2(self.conv_2(out)))
        out = F.relu(self.bn_3(self.conv_3(out)))
        out = torch.max(out, 2, keepdim=True)[0]
        out = out.view(-1, 1024)
        out = F.relu(self.bn_4(self.fc_1(out)))
        out = F.relu(self.bn_5(self.fc_2(out)))
        out = self.fc_3(out)

        #TODO: change numpy functions to torch below
        I = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1,self.k*self.k).repeat(batch_size,1)
        if out.is_cuda:
            I = I.cuda()
        out = out + I
        out = out.view(-1, self.k, self.k)

        return out

class PointNetBase(nn.Module):
    def __init__(self, include_feature_transform=False):
        super(PointNetBase, self).__init__()

        # Transforms
        self.input_transform = SpatialTransformerDimK(3)
        self.include_feature_transform = include_feature_transform
        if include_feature_transform:
            self.feature_transform = SpatialTransformerDimK(64)

        # Shared weight layers
        self.conv_1 = nn.Conv1d(3, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024, 1)

        # Batch norms
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        num_pts = x.size()[2]
        x_input_transformed = self.input_transform(x)
        out = x.transpose(2, 1)
        out = torch.bmm(out, x_input_transformed)
        out = out.transpose(2, 1)
        out = F.relu(self.bn_1(self.conv_1(out)))

        # Transform features if specified to do so
        if self.include_feature_transform:
            transformed_features = self.feature_transform(out)
            out = out.transpose(2, 1)
            out = torch.bmm(out, transformed_features)
            out = out.transpose(2, 1)
        else:
            transformed_features = None
        
        # Continue this b
        point_features = out
        out = F.relu(self.bn_2(self.conv_2(out)))
        out = self.bn_3(self.conv_3(out))
        out = torch.max(out, 2, keepdim=True)[0]
        out = out.view(-1, 1024)
        out = out.view(-1, 1024, 1).repeat(1, 1, num_pts)

        return torch.cat([out, point_features], 1)

class PointNetSegmenter(nn.Module):
    def __init__(self, num_classes, include_feature_transform=False):
        super(PointNetSegmenter, self).__init__()
        self.num_classes = num_classes
        self.include_feature_transform = include_feature_transform
        self.feature_extractor = PointNetBase(include_feature_transform)

        # Shared weight layers
        self.conv_1 = nn.Conv1d(1088, 512, 1)
        self.conv_2 = nn.Conv1d(512, 256, 1)
        self.conv_3 = nn.Conv1d(256, 128, 1)
        self.conv_4 = nn.Conv1d(128, num_classes, 1)

        # Batch norms
        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batch_size = x.size()[0]
        num_pts = x.size()[2]
        out = self.feature_extractor(x)
        out = F.relu(self.bn_1(self.conv_1(out)))
        out = F.relu(self.bn_2(self.conv_2(out)))
        out = F.relu(self.bn_3(self.conv_3(out)))
        out = self.conv_4(out)
        out = out.transpose(2, 1).contiguous()
        out = F.log_softmax(out.view(-1, self.num_classes), dim=-1)
        out  = out.view(batch_size, num_pts, self.num_classes)

        return out

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create sample data
    sim_data = torch.rand(32, 3, 2500)
    sim_data = sim_data.to(device)

    # PointNet features
    point_feat = PointNetBase()
    point_feat.to(device)

    # Segmenter without feature transform
    segmenter_no_feature = PointNetSegmenter(3, include_feature_transform=False)
    segmenter_no_feature.to(device)

    # Segmenter with feature transform
    segmenter_feature = PointNetSegmenter(3, include_feature_transform=True)
    segmenter_feature.to(device)

    # Running data through models
    print('Running sample data through models')
    out = point_feat(sim_data)
    print('point feature output size:', out.size())
    out = segmenter_no_feature(sim_data)
    print('segmenter output size (without feature transform):', out.size())
    out = segmenter_feature(sim_data)
    print('segmenter output size (with feature transform):', out.size())
