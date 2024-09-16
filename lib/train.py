# Modified from https://github.com/open-mmlab/mmgeneration

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../')))

import argparse
import multiprocessing as mp
import os
import os.path as osp
import platform
import time
import warnings

import cv2
import torch
import numpy as np
import random
from torch import distributed as dist

torch.backends.cuda.matmul.allow_tf32 = True

from mmcv import Config

from SUN_dataset import SUN
from ABO_dataset import ABO
from Mvimgnet_dataset import Mvimgnet
from Objaverse_dataset import Objaverse
from Mix_dataset import MixDataset
from get_model import UniGS, UniGS_loss
from train_model import train_model

os.environ['MKL_NUM_THREADS'] = "1"
cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True

def build_dataset(cfg):
    using_channel=cfg.get("using_channel", 14)
    data_split = cfg.get("data_split", "file_paths.json")
    refine_label = cfg.get("refine_label", False)
    load_blip2 = cfg.get("load_blip2", False)

    if cfg.get("type") == "sun":
        train_dataset = SUN(cfg.get("data_dir"), using_channel=using_channel, data_split=data_split, refine_label=refine_label)
        test_dataset = SUN(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split, refine_label=refine_label)
        return train_dataset, test_dataset
    elif cfg.get("type") == "sun_mv":
        train_dataset = SUN(cfg.get("data_dir"), using_channel=using_channel, data_split=data_split, refine_label=refine_label)
        test_dataset = SUN(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split, refine_label=refine_label)
        return train_dataset, test_dataset
    elif cfg.get("type") == "objaverse":
        train_dataset = Objaverse(cfg.get("data_dir"), using_channel=using_channel, data_split=data_split, refine_label=refine_label, load_blip2=load_blip2)
        test_dataset = Objaverse(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split, refine_label=refine_label)
        return train_dataset, test_dataset
    elif cfg.get("type") == "objaverse_sun":
        train_dataset = Objaverse(cfg.get("data_dir"), using_channel=using_channel, data_split=data_split, refine_label=refine_label, 
                                  load_extend=True, load_blip2=load_blip2)
        test_dataset = Objaverse(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split, refine_label=refine_label,
                                 load_extend=True)
        return train_dataset, test_dataset
    elif cfg.get("type") == "abo":
        train_dataset = ABO(cfg.get("data_dir"), using_channel=using_channel, data_split=data_split)
        test_dataset = ABO(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split)
        return train_dataset, test_dataset
    elif cfg.get("type") == "mvimgnet":
        train_dataset = Mvimgnet(cfg.get("data_dir"), using_channel=using_channel, data_split=data_split)
        test_dataset = Mvimgnet(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split)
        return train_dataset, test_dataset
    elif cfg.get("type") == "mix":
        train_dataset = MixDataset(cfg.get("data_dir"), using_channel=using_channel, data_split=data_split)
        test_dataset = MixDataset(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split)
        return train_dataset, test_dataset
    else:
        raise 

def build_model(cfg):
    unigs = UniGS(
        clip_model=cfg["clip_model_path"],
        pointnet_model=cfg["pointnet_model"],
        learning_rate=cfg["learning_rate"],
        pts_channel=cfg["pts_channel"],
        forward_all=cfg["forward_all"],
        model_setting=cfg["model_setting"],
        device="cuda:"+os.environ['RANK'],
    )

    if cfg.resume_from is not None:
        print("Load model from ", cfg.resume_from)
        try:
            if not cfg["model_setting"].get("scratch", False):
                unigs.pointnet.load_state_dict(torch.load(cfg.resume_from, map_location="cpu"))
        except:
            unigs = torch.load(cfg.resume_from, map_location="cpu")
            if not isinstance(unigs, UniGS): unigs = unigs.module

    clip3_loss = UniGS_loss(
        text_weight=cfg["loss_weight"]["text_weight"],
        image_weight=cfg["loss_weight"]["image_weight"],
    )
    return unigs, clip3_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_multi_processes(cfg):
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        mp.set_start_method(mp_start_method)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if ('OMP_NUM_THREADS' not in os.environ and cfg.data_cfg.batch_size > 1):
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and cfg.data_cfg.batch_size > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    cfg.model_cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend='nccl')
        world_size = args.world_size # single machine
        cfg.gpu_ids = range(world_size)

    # create work_dir
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cfg.seed = args.seed
    model, model_loss = build_model(cfg.model_cfg)
    datasets = [*build_dataset(cfg.data_cfg.dataset)]

    train_model(
        model,
        model_loss,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate))


if __name__ == '__main__':
    main()
