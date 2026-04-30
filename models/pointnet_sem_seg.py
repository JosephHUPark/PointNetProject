import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, num_class):
        super(get_model, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=9)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2112,
                nhead=8,
                dim_feedforward=512,
                batch_first=True
            ),
            num_layers=2
        ) # will add a lot of time
        self.conv1 = torch.nn.Conv1d(2112, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.dp1 = nn.Dropout(p=0.3)
        self.dp2 = nn.Dropout(p=0.3)
        
        self.res_conv = nn.Conv1d(512, 256, 1)

    def forward(self, x):
        B, _, N = x.shape

        xyz = x  # keep raw coordinates
        # x must be raw point cloud: [B, 3, N]

        S = 1024
        idx = farthest_point_sampling(xyz, S)  # [B, S]

        # FPS sampling
        x = x[:, :, idx]  # [B, 3, S]
        x, trans, trans_feat = self.feat(x)  # [B, 2112, S]

        x = x.transpose(1, 2)  # [B, S, C]
        x = self.transformer(x) # will add a lot of time (need GPU to feasibly train)
        x = x.transpose(1, 2)  # [B, C, S]
        
        x = F.relu(self.bn1(self.conv1(x)))   # [B,512,S]

        res = self.res_conv(x)                # [B,256,S]

        x = F.relu(self.bn2(self.conv2(x)))   # [B,256,S]
        x = self.dp1(x)
        x = x + res

        x = F.relu(self.bn3(self.conv3(x)))   # [B,128,S]
        x = self.dp2(x)

        x = self.conv4(x)                     # [B,K,S]

        x = x.transpose(2, 1).contiguous()    # [B,S,K]

        return x, trans_feat

def farthest_point_sampling(x, npoint):
    B, _, N = x.shape
    device = x.device

    xyz = x.transpose(1, 2).contiguous()  # [B, N, 3]

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    farthest = torch.randint(0, N, (B,), device=device)

    batch_indices = torch.arange(B, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest

        centroid = xyz[batch_indices, farthest].view(B, 1, 3)

        dist = torch.sum((xyz - centroid) ** 2, dim=-1)

        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = torch.max(distance, dim=1)[1]

    return centroids

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.cross_entropy(pred, target, weight=weight, label_smoothing=0.1)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__ == '__main__':
    model = get_model(13)
    xyz = torch.rand(12, 3, 2048)
    out, _ = model(xyz)
