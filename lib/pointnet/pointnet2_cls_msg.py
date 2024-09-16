import torch.nn as nn
import torch.nn.functional as F
from pointnet.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True,pts_channel=14,forward_all=True,**kwargs):
        super(get_model, self).__init__()
        in_channel = pts_channel - 3 if normal_channel else pts_channel

        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])

        if not forward_all:
            self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 512], True)
        else:
            self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
            
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
    
    def get_parameters(self):
        return self.parameters()
    
    def forward(self, xyz, forward_all=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        # xyz [batch, 14, 1024]

        l1_xyz, l1_points = self.sa1(xyz, norm) # [batch, 3, 1024], [batch, 320, 1024]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # [batch, 3, 128], [batch, 640, 128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # [batch, 3, 1], [batch, 1024, 1]
        if not forward_all: return l3_points.view(B, -1), None

        x = l3_points.view(B, 1024) #[batch, 1024]
        x = self.drop1(F.relu(self.bn1(self.fc1(x)))) # [batch, 512]
        x = self.drop2(F.relu(self.bn2(self.fc2(x)))) # [batch, 256]
        x = self.fc3(x) #  # [batch, 512]
        # x = F.log_softmax(x, -1) # auxiliary loss

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


