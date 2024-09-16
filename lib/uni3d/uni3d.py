import torch
import timm
import numpy as np
from torch import nn
from uni3d.losses import Uni3d_Text_Image_Loss

from uni3d.point_encoder import PointcloudEncoder, ParallelPointcloudEncoder, ControlPointcloudEncoder, ConcatPointcloudEncoder

class Uni3D(nn.Module):
    def __init__(self, point_encoder, load_rgb=False):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder
        self.load_rgb = load_rgb
    
    def get_parameters(self):
        return self.point_encoder.get_parameters()

    def encode_pc(self, pc):
        xyz = pc[:,:,:3].contiguous()
        color = pc[:,:,3:].contiguous()
        pc_feat = self.point_encoder(xyz, color)
        return pc_feat

    def forward(self, pc, **kwargs): # [24, 14, 1024]
        xyz = pc[:,:3,:].transpose(1,2).contiguous()
        color = pc[:,3:6,:].transpose(1,2).contiguous()

        if hasattr(self, "load_rgb") and self.load_rgb: other = pc[:,3:,:].transpose(1,2).contiguous()
        else: other = pc[:,6:,:].transpose(1,2).contiguous()

        pc_feat = self.point_encoder(xyz, color, other)

        return pc_feat, None

    def forward_all(self, pc, text, image):
        text_embed_all = text
        image_embed = image   
        pc_embed = self.encode_pc(pc)
        return {'text_embed': text_embed_all,
                'pc_embed': pc_embed,
                'image_embed': image_embed,
                'logit_scale': self.logit_scale.exp()}

def get_filter_loss(args):
    return Uni3d_Text_Image_Loss()

def get_metric_names(model):
    return ['loss', 'uni3d_loss', 'pc_image_acc', 'pc_text_acc']

def create_uni3d(args, load_pretrained=False, model_type="", load_rgb=False, scratch=False):
    # create transformer blocks for point cloud via timm
    point_transformer = timm.create_model(args["pc_model"], checkpoint_path=args["pretrained_pc"], drop_path_rate=args["drop_path_rate"])

    # create whole point cloud encoder
    if model_type == "concat": point_encoder = ConcatPointcloudEncoder(point_transformer, args, load_pretrained=load_pretrained, ckpt_path=args["ckpt_path"])
    elif model_type == "parallel": point_encoder = ParallelPointcloudEncoder(point_transformer, args, load_pretrained=load_pretrained, ckpt_path=args["ckpt_path"], load_rgb=load_rgb)
    elif model_type == "control": point_encoder = ControlPointcloudEncoder(point_transformer, args, load_pretrained=load_pretrained, ckpt_path=args["ckpt_path"], load_rgb=load_rgb)
    else: point_encoder = PointcloudEncoder(point_transformer, args, load_pretrained=load_pretrained, ckpt_path=args["ckpt_path"], scratch=scratch)

    # uni3d model
    model = Uni3D(point_encoder, load_rgb)
    return model

def create_uni3d_test():  
    # create transformer blocks for point cloud via timm
    # pc_model="eva_giant_patch14_560.m30m_ft_in22k_in1k"

    # # Giant
    # pc_model="eva_giant_patch14_560"
    # pc_feat_dim=1408
    # # Large
    # pc_model="eva02_large_patch14_448"
    # pc_feat_dim=1024
    # # base
    # pc_model="eva02_base_patch14_448"
    # pc_feat_dim=768
    # # small
    # pc_model="eva02_small_patch14_224"
    # pc_feat_dim=384
    # # tiny
    # pc_model="eva02_tiny_patch14_224"
    # pc_feat_dim=192

    pc_model="eva02_base_patch14_448"
    point_transformer = timm.create_model(pc_model,drop_path_rate=0.2)

    class arg():
        pc_feat_dim=768 #768 1408
        embed_dim=512 #512 1024
        group_size=64 #32
        num_group=512
        pc_encoder_dim=256
        patch_dropout=0.5
        in_channel=14

    args = arg()

    # create whole point cloud encoder
    point_encoder = PointcloudEncoder(point_transformer, args)

    # uni3d model
    model = Uni3D(point_encoder=point_encoder)
    return model


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    model = create_uni3d_test().cuda()

    pc_data = torch.rand((2,1024,14), device="cuda")
    text_embed = torch.rand((2,512), device="cuda")
    image_embed = torch.rand((2,512), device="cuda")
    outputs = model(pc_data, text_embed, image_embed)

    print("pc_embed", outputs["pc_embed"].shape)