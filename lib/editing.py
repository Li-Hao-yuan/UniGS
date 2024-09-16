import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..')))

import json
import torch
import torch.nn as nn
from mmcv import Config
from tqdm import tqdm
import argparse
import numpy as np
import random
from plyfile import PlyData, PlyElement
import clip

from get_model import UniGS

''' 
CUDA_VISIBLE_DEVICES=0 python /path/to/your/unigs/lib/editing.py \
    /path/to/your/unigs/work_dirs/paper/objaverse_retrive/1e-4/objaverse-100k_clip3-parallel-rgb/objaverse_uni3d.py \
    --resume-from /path/to/your/unigs/work_dirs/paper/objaverse_retrive/1e-4/objaverse-100k_clip3-parallel-rgb/model_epoch50.pth \
    --object /path/to/your/gaussian-splatting/output/test/point_cloud/iteration_3000/point_cloud_old.ply

'''
class module_requires_grad:
    def __init__(self, module, requires_grad=True):
        self.module = module
        self.requires_grad = requires_grad
        self.prev = []

    def __enter__(self):
        for p in self.module.parameters():
            self.prev.append(p.requires_grad)
            p.requires_grad = self.requires_grad

    def __exit__(self, exc_type, exc_value, traceback):
        for p, r in zip(self.module.parameters(), self.prev):
            p.requires_grad = r

######

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--object', help='xyz')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def build_model(cfg):
    unigs = UniGS(
        clip_model=cfg["clip_model_path"],
        pointnet_model=cfg["pointnet_model"],
        learning_rate=cfg["learning_rate"],
        pts_channel=cfg["pts_channel"],
        forward_all=cfg["forward_all"],
        model_setting=cfg["model_setting"],
    )
    return unigs

def load_checkpoint(cfg):

    unigs = UniGS(
            clip_model=cfg["model_cfg"]["clip_model_path"],
            pointnet_model=cfg["model_cfg"]["pointnet_model"],
            learning_rate=cfg["model_cfg"]["learning_rate"],
            pts_channel=cfg["model_cfg"]["pts_channel"],
            forward_all=cfg["model_cfg"]["forward_all"],
            model_setting=cfg["model_cfg"]["model_setting"],
        )
    if not cfg["model_cfg"]["model_setting"].get("scratch", False):
        unigs.pointnet.load_state_dict(torch.load(cfg.resume_from, map_location="cpu"))

    return unigs

def load_ply(path, max_sh_degree=0):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # _xyz: [1024,3]
    # _features_dc: [1024,1,3]
    # _features_rest: [1024,15,sh_dgree]
    # _opacity: [1024,1]
    # _scaling: [1024,3]
    # _rotation: [1024,4]

    _xyz = torch.tensor(xyz, dtype=torch.float)
    num_pts = _xyz.shape[0]
    _features_dc = torch.tensor(features_dc, dtype=torch.float).transpose(1, 2).contiguous().reshape(num_pts, -1) # 3,1 -> 1,3 -> 3
    _features_rest = torch.tensor(features_extra, dtype=torch.float).transpose(1, 2).contiguous().reshape(num_pts, -1)
    _opacity = torch.tensor(opacities, dtype=torch.float)
    _scaling = torch.tensor(scales, dtype=torch.float)
    _rotation = torch.tensor(rots, dtype=torch.float)

    pts_feature = torch.cat(
        (_xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation), dim=1
    )

    return pts_feature

def save_ply(data, save_path):

    # _xyz: [1024,3]
    # _features_dc: [1024,1,3]
    # _features_rest: [1024,15,3]
    # _opacity: [1024,1]
    # _scaling: [1024,3]
    # _rotation: [1024,4]

    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        # for i in range(45):
        #     l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l
    
    xyz = data[:,:3].data.numpy()
    f_dc = data[:,3:6].reshape((-1,1,3)).transpose(1, 2).flatten(start_dim=1).data.numpy() # 1,3 -> 3,1 -> 3
    opacities = data[:,6:7].data.numpy()
    scale = data[:,7:10].data.numpy()
    rotation = data[:,10:14].data.numpy()
    normals = np.zeros_like(xyz)

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(save_path)

def editiing_color(model, ply_feature, text):
    ply_feature = ply_feature.transpose(0,1).contiguous().unsqueeze(0).cuda()
    ply_feature[:,3:] = torch.tanh(ply_feature[:,3:])

    xyz = ply_feature[:,:3,:]
    color = ply_feature[:,3:6,:]
    others = ply_feature[:,6:,:]

    model.eval()

    color_param = torch.nn.Parameter(color).requires_grad_()
    optimizer = torch.optim.AdamW([color_param], lr=1e-2)

    text = clip.tokenize(text).cuda()
    text_features = model.encode_text(text).to(torch.float32)
    text_features = nn.functional.normalize(text_features, dim=1, p=2)

    with module_requires_grad(model, False):
        for i in range(100):

            optimizer.zero_grad()
            gaussian_features, _ = model.encode_gaussian(torch.cat((xyz,color_param,others),dim=1))
            gaussian_features = nn.functional.normalize(gaussian_features, dim=1, p=2)

            loss = 1 - torch.cosine_similarity(gaussian_features, text_features)

            loss.backward(retain_graph=True)
            optimizer.step()

            print("Count: %d | Loss: %.4f"%(i, loss.item()))

    edited_plf_feature = torch.cat((xyz, color_param.data, others), dim=1).transpose(1,2).contiguous().squeeze(0)
    edited_plf_feature[:,3:] = torch.arctanh(edited_plf_feature[:,3:].clamp(-1+1e-4,1-1e-4))
    return edited_plf_feature.cpu()

def editing_by_prompt(model, cfg, text):
    # _xyz: [1024,3]
    # _features_dc: [1024,1,3]
    # _features_rest: [1024,15,sh_dgree]
    # _opacity: [1024,1]
    # _scaling: [1024,3]
    # _rotation: [1024,4]

    ply_feature = load_ply(cfg.object)
    edited_3dgs =  editiing_color(model, ply_feature, text)

    save_ply(edited_3dgs, os.path.join(cfg.work_dir, "edited.ply"))

    

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else: cfg.work_dir = 'work_dirs/editing'
    os.makedirs(cfg.work_dir, exist_ok=True)

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cfg.seed = args.seed
    cfg.object = args.object
    device = torch.device("cuda:"+str(args.gpu_id))

    model = load_checkpoint(cfg).to(device)
    editing_by_prompt(model, cfg, "red")


if __name__ == '__main__':
    main()