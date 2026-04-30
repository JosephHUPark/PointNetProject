import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
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
        x, trans, trans_feat = self.feat(x)
        x = x.transpose(1, 2)
        
        x = self.transformer(x).transpose(1, 2)
        
        batchsize = x.size()[0]
        n_pts = x.size()[2]
    
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))   # [B,512,N]
    
        res = self.res_conv(x)                # [B,256,N]
        x = F.relu(self.bn2(self.conv2(x)))   # [B,256,N]
        x = self.dp1(x)
        x = x + res
    
        x = F.relu(self.bn3(self.conv3(x)))   # [B,128,N]
        x = self.dp2(x)
        x = self.conv4(x)
    
        x = x.transpose(2,1).contiguous()
        x = x.view(batchsize, n_pts, self.k)
    
        return x, trans_feat

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
    (model(xyz))
