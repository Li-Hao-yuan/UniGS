import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from pointnet.transformer import TransformerEncoder

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True,pts_channel=14,forward_all=True,model_setting=None,**kwargs):
        super(get_model, self).__init__()
        in_channel = pts_channel - 3 if normal_channel else pts_channel

        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]]) # 512, 1024
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])

        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
            
        self.fc1 = nn.Linear(1024, 512)
        # self.fc1 = nn.Linear(2048, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)

        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

        # trans
        self.encoder_dims = model_setting["encoder_dims"]
        self.trans_dim = model_setting["trans_dim"]

        self.trans = TransformerEncoder(self.trans_dim)
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.norm = nn.LayerNorm(self.trans_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

    def get_parameters(self):
        return self.parameters()
    
    def forward(self, xyz, forward_all=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm) # [batch, 3, 512], [batch, 320, 512]

        l1_center = l1_xyz.transpose(1,2) # [batch, 512, 3]
        l1_feature = self.reduce_dim(l1_points.transpose(1,2)) # [abtch, 512, 320]
        cls_tokens = self.cls_token.expand(l1_feature.size(0), -1, -1)
        cls_pos = self.cls_token.expand(l1_feature.size(0), -1, -1)
        pos = self.pos_embed(l1_center)
        l1_feature = torch.cat((cls_tokens, l1_feature), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        trans_feat = self.trans(l1_feature, pos)
        trans_feat = self.norm(trans_feat)
        concat_f = torch.cat([trans_feat[:, 0], trans_feat[:, 1:].max(1)[0]], dim=-1) # [32, 640]

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # [batch, 3, 128], [batch, 640, 128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # [batch, 3, 1], [batch, 1024, 1]
        if not forward_all: return l3_points.view(B, -1), None

        x = l3_points.view(B, 1024) + concat_f #[batch, 1024]
        # x = torch.cat((l3_points.view(B, 1024), concat_f), dim=-1)

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))# + trans_feat[:, 0] # [batch, 512]
        x = self.drop2(F.relu(self.bn2(self.fc2(x)))) # [batch, 256]
        x = self.fc3(x)# + trans_feat[:, 0]#  # [batch, 512]
        # x = F.log_softmax(x, -1) # auxiliary loss

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


