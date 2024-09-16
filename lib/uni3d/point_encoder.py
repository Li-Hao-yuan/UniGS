import torch
from torch import nn, einsum
# from pointnet2_ops import pointnet2_utils

import logging
import copy
from einops import rearrange, repeat
from inspect import isfunction
from typing import Optional, Any

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    # zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    # fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    # fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()

    fps_idx = farthest_point_sample(data, number) # [B, npoint, C]
    fps_data = index_points(data, fps_idx)

    return fps_data

# https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py 
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist    

def get_this_state_dict(path):
    state_dcit = torch.load(path)["module"]
    new_state_dcit = {}
    for key in state_dcit.keys():
        if "point_encoder" in key:
            new_state_dcit[ key[14:] ] = state_dcit[key]
    del state_dcit
    return new_state_dcit

class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info("patch dropout prob is {}".format(prob))

    def forward(self, x):
        # if not self.training or self.prob == 0.:
        #     return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        '''
            input: B N C
            ---------------------------
            output: B G M C
            center : B G C
        '''
        batch_size, num_points, _ = xyz.shape
        channel = color.shape[-1]
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, channel).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features

class Encoder(nn.Module):
    def __init__(self, encoder_channel, in_channel=6):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, -1)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

class PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, args, load_pretrained=False,
                 ckpt_path="/path/to/your/unigs/cache/uni3d-S.pt", scratch=False):
        super().__init__()

        # from easydict import EasyDict
        self.trans_dim = args["pc_feat_dim"] # 768
        self.embed_dim = args["embed_dim"] # 512
        self.group_size = args["group_size"] # 32
        self.num_group = args["num_group"] # 512
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dim =  args["pc_encoder_dim"] # 256
        self.encoder = Encoder(encoder_channel = self.encoder_dim, in_channel=args["in_channel"])
        self.in_channel = args["in_channel"]
       
        # bridge encoder and transformer
        self.encoder2trans = nn.Linear(self.encoder_dim,  self.trans_dim)
        
        # bridge transformer and clip embedding
        self.trans2embed = nn.Linear(self.trans_dim,  self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.scratch = scratch

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(args["patch_dropout"]) if args["patch_dropout"] > 0. else nn.Identity()
        self.visual = point_transformer
        
        self.load_pretrained = load_pretrained
        if self.load_pretrained:
            self.load_state_dict(get_this_state_dict(ckpt_path), strict=False)
            self.eval()
            for _, param in self.named_parameters():
                param.requires_grad = False

            if not scratch:
                self.conv1 = torch.nn.Sequential(
                    nn.Conv1d(1024, 64, 3, 1, 1),
                    nn.SiLU(),
                    nn.Conv1d(64, 8, 3, 1, 1),
                )
                self.mlp1 = nn.Linear(64, 512)
                self.mlp2 = nn.Linear(1024, 512)
    
    def get_parameters(self):
        if self.load_pretrained:
            if hasattr(self, "scratch") and self.scratch: return []
            else: return list(self.conv1.parameters())+list(self.mlp1.parameters())+list(self.mlp2.parameters())
        else:
            return self.parameters()
    
    def encode_feature(self, pts, colors):
        # divide the point cloud in the same form. This is important
        _, center, features = self.group_divider(pts, colors)

        # encoder the input cloud patches
        group_input_tokens = self.encoder(features)  #  B G N, [batch, 512, 512]

        group_input_tokens = self.encoder2trans(group_input_tokens) # [batch, 512, 768]
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1) # [batch, 1, 768]
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)   # [batch, 1, 768]
        # add pos embedding
        pos = self.pos_embed(center) # [batch, 512, 768], [batch, 512, 3]
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = x + pos
        
        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)

        x = self.visual.pos_drop(x)

        return x

    def forward(self, pts, colors, other):

        if hasattr(self, "in_channel") and colors.shape[2] != (self.in_channel-3): colors = other
        x = self.encode_feature(pts, colors)

        # ModuleList not support forward
        for i, blk in enumerate(self.visual.blocks):
            x = blk(x)
        x = self.visual.norm(x[:, 0, :]) # [32, 768] -> [32, 768]
        x = self.visual.fc_norm(x) # [32, 768] -> [32, 768]

        x = self.trans2embed(x) # [32, 768] -> [32, 1024]

        if self.load_pretrained:
            if hasattr(self, "scratch") and self.scratch: return x
            else:
                other = self.conv1(other).view(-1, 64)
                # other = self.conv1(torch.cat((pts,other),dim=-1)).view(-1, 88)
                x = self.mlp1(other) + self.mlp2(x)

        return x

class ParallelPointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, args, load_pretrained=False, 
                 ckpt_path="/path/to/your/unigs/cache/uni3d-S.pt", load_rgb=False):
        super().__init__()
        # from easydict import EasyDict
        self.trans_dim = args["pc_feat_dim"] # 768
        self.embed_dim = args["embed_dim"] # 512
        self.group_size = args["group_size"] # 32
        self.num_group = args["num_group"] # 512
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dim =  args["pc_encoder_dim"] # 256
        self.encoder = Encoder(encoder_channel = self.encoder_dim, in_channel=6)

        if load_rgb: self.gs_encoder = Encoder(encoder_channel = self.encoder_dim, in_channel=14)
        else: self.gs_encoder = Encoder(encoder_channel = self.encoder_dim, in_channel=11)
       
        # bridge encoder and transformer
        self.encoder2trans = nn.Linear(self.encoder_dim,  self.trans_dim)
        
        # bridge transformer and clip embedding
        self.trans2embed = nn.Linear(self.trans_dim,  self.embed_dim)
        self.out = nn.Linear(self.embed_dim, 512)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(args["patch_dropout"]) if args["patch_dropout"] > 0. else nn.Identity()

        self.visual = point_transformer
        self.gs_vit = copy.deepcopy(point_transformer)

        cross_attn = []
        for _ in range(len(self.visual.blocks)):
            if XFORMERS_IS_AVAILBLE:
                cross_attn.append(MemoryEfficientCrossAttention(self.trans_dim, self.trans_dim))
            else: cross_attn.append(CrossAttention(self.trans_dim, self.trans_dim))
        self.cross_attn = nn.ModuleList(cross_attn)
            
        self.load_pretrained = load_pretrained
        if self.load_pretrained:
            self.load_state_dict(get_this_state_dict(ckpt_path), strict=False)
            self.visual.eval()
            for _, param in self.visual.named_parameters():
                param.requires_grad = False
    
    def get_parameters(self):    
    
        if self.load_pretrained:
            parameters = []
            for name, param in self.named_parameters():
                if "visual." in name: continue
                parameters.append(param)
            return parameters
        else:
            return self.parameters()

    def encode_feature(self, pts, colors, encode_pt=True):
        # divide the point cloud in the same form. This is important
        _, center, features = self.group_divider(pts, colors)

        # encoder the input cloud patches
        if encode_pt:
            group_input_tokens = self.encoder(features)  #  B G N, [batch, 512, 512]
        else: group_input_tokens = self.gs_encoder(features)  #  B G N, [batch, 512, 512]

        group_input_tokens = self.encoder2trans(group_input_tokens) # [batch, 512, 768]
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1) # [batch, 1, 768]
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)   # [batch, 1, 768]
        # add pos embedding
        pos = self.pos_embed(center) # [batch, 512, 768], [batch, 512, 3]
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = x + pos
        
        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)

        if encode_pt:
            x = self.visual.pos_drop(x)
        else: x = self.gs_vit.pos_drop(x)

        return x

    def forward(self, pts, colors, others):
        
        pt_x = self.encode_feature(pts, colors, True)
        gs_x = self.encode_feature(pts, others, False)

        # ModuleList not support forward
        for pt_blk, gs_blk, cross_attn in zip(self.visual.blocks, self.gs_vit.blocks, self.cross_attn):
            gs_x = gs_x + cross_attn(gs_x, pt_x)
            pt_x = pt_blk(pt_x)
            gs_x = gs_blk(gs_x)

        pt_x = self.visual.norm(pt_x[:, 0, :])
        gs_x = self.gs_vit.norm(gs_x[:, 0, :])

        pt_x = self.visual.fc_norm(pt_x)
        gs_x = self.gs_vit.fc_norm(gs_x)

        pt_x = self.trans2embed(pt_x)
        gs_x = self.trans2embed(gs_x)

        return self.out(pt_x+gs_x)


class ControlPointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, args, load_pretrained=False, 
                 ckpt_path="/path/to/your/unigs/cache/uni3d-B.pt", load_rgb=False):
        super().__init__()
        self.trans_dim = args["pc_feat_dim"] # 768
        self.embed_dim = args["embed_dim"] # 512
        self.group_size = args["group_size"] # 32
        self.num_group = args["num_group"] # 512
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dim =  args["pc_encoder_dim"] # 256

        if load_rgb: self.encoder = Encoder(encoder_channel = self.encoder_dim, in_channel=14)
        else: self.encoder = Encoder(encoder_channel = self.encoder_dim, in_channel=11)
       
        # bridge encoder and transformer
        self.encoder2trans = nn.Linear(self.encoder_dim,  self.trans_dim)
        
        # bridge transformer and clip embedding
        self.trans2embed = nn.Linear(self.trans_dim,  self.embed_dim)

        self.out_conv = zero_module(conv_nd(1,1,1,1,padding=0))
        self.out_linear = nn.Linear(self.embed_dim*2, 512)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(args["patch_dropout"]) if args["patch_dropout"] > 0. else nn.Identity()

        self.point_encoder = PointcloudEncoder(copy.deepcopy(point_transformer), args, load_pretrained, ckpt_path)
        self.gs_vit = point_transformer
        for _, param in self.gs_vit.named_parameters():
            param.requires_grad = True

        cross_attn = []
        for _ in range(len(self.gs_vit.blocks)):
            if XFORMERS_IS_AVAILBLE:
                cross_attn.append(MemoryEfficientCrossAttention(self.trans_dim, self.trans_dim))
            else: cross_attn.append(CrossAttention(self.trans_dim, self.trans_dim))
        self.cross_attn = nn.ModuleList(cross_attn)

        self.load_pretrained = load_pretrained
        if self.load_pretrained:
            self.load_state_dict(torch.load(ckpt_path)["module"], strict=False)
            self.point_encoder.eval()
            for _, param in self.point_encoder.named_parameters():
                param.requires_grad = False
    
    def get_parameters(self):
    
        if self.load_pretrained:
            parameters = []
            for name, param in self.named_parameters():
                if "point_encoder." in name: continue
                parameters.append(param)
            return parameters
        else:
            return self.parameters()
    
    def encode_feature(self, pts, others):
        # divide the point cloud in the same form. This is important
        _, center, features = self.group_divider(pts, others)

        # encoder the input cloud patches
        group_input_tokens = self.encoder(features)  #  B G N, [batch, 512, 512]

        group_input_tokens = self.encoder2trans(group_input_tokens) # [batch, 512, 768]
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1) # [batch, 1, 768]
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)   # [batch, 1, 768]
        # add pos embedding
        pos = self.pos_embed(center) # [batch, 512, 768], [batch, 512, 3]
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = x + pos
        
        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)

        x = self.gs_vit.pos_drop(x)

        return x

    def forward(self, pts, colors, others):
        
        pt_x = self.point_encoder.encode_feature(pts, colors)
        gs_x = self.encode_feature(pts, others) + pt_x

        # ModuleList not support forward
        for pt_blk, gs_blk, cross_attn in zip(self.point_encoder.visual.blocks, self.gs_vit.blocks, self.cross_attn):
            # gs_x = gs_blk(gs_x)
            # pt_x = pt_blk(pt_x + zero_conv(gs_x))
            # pt_x = pt_blk(pt_x + cross_attn(gs_x))
            # pt_x = pt_blk(pt_x + zero_conv(cross_attn(gs_x)))

            gs_x = gs_x + cross_attn(gs_x, pt_x)
            pt_x = pt_blk(pt_x)
            gs_x = gs_blk(gs_x)

        pt_x = self.point_encoder.visual.norm(pt_x[:, 0, :])
        gs_x = self.gs_vit.norm(gs_x[:, 0, :])

        pt_x = self.point_encoder.visual.fc_norm(pt_x)
        gs_x = self.gs_vit.fc_norm(gs_x)

        pt_x = self.point_encoder.trans2embed(pt_x)
        gs_x = self.trans2embed(gs_x)

        output = self.out_linear(torch.cat((pt_x, gs_x),dim=-1))

        return output
    
class ConcatPointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, args, load_pretrained=False, 
                 ckpt_path="/path/to/your/unigs/cache/uni3d-S.pt"):
        super().__init__()
        
        self.load_pretrained = load_pretrained
        self.point_encoder = PointcloudEncoder(copy.deepcopy(point_transformer), args, False, ckpt_path)

        args["in_channel"] = 11
        self.gs_encoder = PointcloudEncoder(point_transformer, args, False, ckpt_path)
        self.out_mlp = nn.Linear(2048, 512)
    
        if self.load_pretrained:
            self.load_state_dict(torch.load(ckpt_path)["module"], strict=False)
            self.point_encoder.eval()
            for _, param in self.point_encoder.named_parameters():
                param.requires_grad = False

    def get_parameters(self):
        if self.load_pretrained:
            parameters = []
            for name, param in self.named_parameters():
                if "point_encoder." in name: continue
                parameters.append(param)
            return parameters
        else:
            return self.parameters()


    def forward(self, pts, colors, others):
        
        pt_x = self.point_encoder(pts, colors, None)
        gs_x = self.gs_encoder(pts, others, None)
        
        output = self.out_mlp(torch.cat((pt_x, gs_x),dim=-1))


        return output



