import torch
import torch.nn as nn
import torch.nn.functional as F

class TNetK(nn.Module):
    def __init__(self, k=64):
        super(TNetK, self).__init__()

        self.core = PointNetCore(k)

        # Fully connected layers
        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, k**2)

        # Batch norms
        self.bn_1 = nn.InstanceNorm1d(512)
        self.bn_2 = nn.InstanceNorm1d(256)

        # Other
        self.k = k
        # self.maxpool = nn.MaxPool1d()

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.core(x)
        print(out.shape)
        out = F.relu(self.bn_1(self.fc_1(out)))
        out = F.relu(self.bn_2(self.fc_2(out)))
        out = self.fc_3(out)
        I = torch.eye(self.k).flatten()
        if out.is_cuda:
            I = I.cuda()
        out = out + I
        out = out.view(batch_size, self.k, self.k)
        
        return out

class PointNetCore(nn.Module):
    def __init__(self, input_dim):
        super(PointNetCore, self).__init__()

        # Shared weight layers
        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024, 1)

        # Batch norms
        self.bn_1 = nn.InstanceNorm1d(64)
        self.bn_2 = nn.InstanceNorm1d(128)
        self.bn_3 = nn.InstanceNorm1d(1024)

    def forward(self, x):
        out = F.relu(self.bn_1(self.conv_1(x)))
        out = F.relu(self.bn_2(self.conv_2(out)))
        out = F.relu(self.bn_3(self.conv_3(out)))
        out = torch.max(out, 2, keepdim=True)[0] # Max pool across points
        out = out.view(-1, 1024)

        return out

class PointNetTransformationHead(nn.Module):
    def __init__(self):
        super(PointNetTransformationHead, self).__init__()

        # Create transformation networks
        self.input_transform_net = TNetK(3)
        self.feature_transform_net = TNetK(64)

        # Shared layers
        self.conv_1 = nn.Conv1d(3, 64, 1)

        # Batch norms
        self.bn_1 = nn.InstanceNorm1d(64)

    def forward(self, x):
        input_transform = self.input_transform_net(x)
        out = x.transpose(2, 1)
        out = torch.bmm(out, input_transform)
        out = out.transpose(2, 1)
        out = F.relu(self.bn_1(self.conv_1(out)))
        feature_transform = self.feature_transform_net(out)
        out = out.transpose(2, 1)
        out = torch.bmm(out, feature_transform)
        out = out.transpose(2, 1) # These are the point features

        return out, input_transform, feature_transform

class BasePointNet(nn.Module):
    def __init__(self):
        super(BasePointNet, self).__init__()
        self.transformation_head = PointNetTransformationHead()
        self.core = PointNetCore(64)

    def forward(self, x):
        point_features, input_transform, feature_transform = self.transformation_head(x)
        global_features = self.core(point_features)

        return point_features, global_features, input_transform, feature_transform

class PointNetSegmenter(nn.Module):
    def __init__(self, num_classes):
        super(PointNetSegmenter, self).__init__()
        self.point_net_base = BasePointNet()

        self.num_classes = num_classes

        # Shared weights
        self.conv_1 = nn.Conv1d(1088, 512, 1)
        self.conv_2 = nn.Conv1d(512, 256, 1)
        self.conv_3 = nn.Conv1d(256, 128, 1)
        self.conv_4 = nn.Conv1d(128, self.num_classes, 1)

        # Batch norms
        self.bn_1 = nn.InstanceNorm1d(512)
        self.bn_2 = nn.InstanceNorm1d(256)
        self.bn_3 = nn.InstanceNorm1d(128)

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[2]
        point_features, global_features, input_transform, feature_transform = self.point_net_base(x)
        global_features = global_features.unsqueeze(2).repeat(1, 1, num_points)
        out = torch.cat((point_features, global_features), dim=1)
        out = F.relu(self.bn_1(self.conv_1(out)))
        out = F.relu(self.bn_2(self.conv_2(out)))
        out = F.relu(self.bn_3(self.conv_3(out)))
        out = self.conv_4(out) # Logits
        out = out.transpose(2, 1).contiguous() #NOTE: I didn't actually check the next lines (including this one)
        out = F.log_softmax(out.view(-1, self.num_classes), dim=-1)
        out = out.view(batch_size, num_points, self.num_classes)
        out = out.transpose(2, 1)

        if self.training:
            return out, input_transform, feature_transform
        else:
            return out

def orthogonal_loss(transformation_matrices):
    """Orthogonal matrix loss

    Parameters
    ----------
    transformation_matrices: torch.Tensor
        Batch of transformation matricies

    Returns
    torch.tensor
        Orthogonal matrix loss
    """

    dim = transformation_matrices.shape[1]
    I = torch.eye(dim)[None, :, :]
    if transformation_matrices.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(transformation_matrices, transformation_matrices.transpose(2, 1)) - I, dim=(1, 2)))

    return loss

if __name__ == '__main__':
    batch_size = 3
    num_points = 5
    sim_3d_data = torch.randn(batch_size, 3, num_points)
    print(f"3d input shape: {sim_3d_data.shape}")
    sim_64d_data = torch.randn(batch_size, 64, num_points)
    print(f"64d input shape: {sim_64d_data.shape}")

    # 3d transformation net
    point_transform_3d = TNetK(3).eval()
    out = point_transform_3d(sim_3d_data)
    print(f"\npoint transform matrix shape: {out.shape}")
    print(f"orthoganal loss: {orthogonal_loss(out)}")

    # 64d transformation net
    point_transform_64d = TNetK(64).eval()
    out = point_transform_64d(sim_64d_data)
    print(f"\npoint transform matrix shape: {out.shape}")
    print(f"orthoganal loss: {orthogonal_loss(out)}")

    # Core network
    core_network = PointNetCore(3).eval()
    out = core_network(sim_3d_data)
    print(f"\ncore network output shape: {out.shape}")

    # Transformation head
    transformation_head = PointNetTransformationHead().eval()
    out, input_transform, feature_transform = transformation_head(sim_3d_data)
    print(f"\ntransformation head output shape: {out.shape}")
    print(f"point transform loss: {orthogonal_loss(input_transform)}")
    print(f"feature transform loss: {orthogonal_loss(feature_transform)}")

    # Base pointnet
    base_net = BasePointNet().eval()
    point_features, global_features, input_transform, feature_transform = base_net(sim_3d_data)
    print(f"\npoint_feature shape: {point_features.shape}")
    print(f"global_feature shape: {global_features.shape}")
    print(f"point transform loss: {orthogonal_loss(input_transform)}")
    print(f"feature transform loss: {orthogonal_loss(feature_transform)}")

    # # Segmenter
    segmenter = PointNetSegmenter(3).train()
    out, input_transform, feature_transform = segmenter(sim_3d_data)
    print(f"output shape: {out.shape}")
    print(f"point transform loss: {orthogonal_loss(input_transform)}")
    print(f"feature transform loss: {orthogonal_loss(feature_transform)}")
    segmenter.eval()
    out = segmenter(sim_3d_data)
    print(f"output shape: {out.shape}")
