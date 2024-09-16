# Modified from https://github.com/open-mmlab/mmgeneration

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../')))

import argparse
import json

import cv2
import clip
import math
import torch
import numpy as np
import random
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True

from mmcv import Config
from tqdm import tqdm

from get_model import UniGS
from SUN_dataset import SUN
from Objaverse_dataset import Objaverse
from Mvimgnet_dataset import Mvimgnet
from ABO_dataset import ABO
from base_dataset import BaseDataset
from torch.utils.data import DataLoader
from prettytable import PrettyTable




def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--k', type=str, default="1")
    parser.add_argument('--task', type=str, default="")
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--test_dataset', type=str, default="")
    parser.add_argument('--data_split', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--update_label', action="store_true", default=False)
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

def build_dataset(cfg, update_label):
    using_channel=cfg.get("using_channel", 14)
    data_split = cfg.get("data_split", "file_paths.json")
    refine_label = cfg.get("refine_label", False)

    if cfg.get("type") in ["sun","sunrgbd"]:
        test_dataset = SUN(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split, refine_label=refine_label)
        return test_dataset
    elif cfg.get("type") == "sun_mv":
        test_dataset = SUN(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split, refine_label=refine_label)
        return test_dataset
    elif cfg.get("type") == "objaverse":
        test_dataset = Objaverse(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split, update_label=update_label)
        return test_dataset
    elif cfg.get("type") == "objaverse_sun":
        test_dataset = Objaverse(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split, load_sun=True)
        return test_dataset
    elif cfg.get("type") == "abo":
        test_dataset = ABO(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split)
        return test_dataset
    elif cfg.get("type") == "mvimgnet":
        test_dataset = Mvimgnet(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split, update_label=update_label)
        return test_dataset
    else:
        test_dataset = BaseDataset(cfg.get("data_dir"), is_train=False, using_channel=using_channel, data_split=data_split)
        return test_dataset

def load_checkpoint(cfg):
    # model = torch.load(cfg.resume_from, map_location="cpu")
    # if not isinstance(model, UniGS): model = model.module

    # unigs = UniGS(
    #         clip_model=cfg["model_cfg"]["clip_model_path"],
    #         pointnet_model=cfg["model_cfg"]["pointnet_model"],
    #         learning_rate=cfg["model_cfg"]["learning_rate"],
    #         pts_channel=cfg["model_cfg"]["pts_channel"],
    #         forward_all=cfg["model_cfg"]["forward_all"],
    #         model_setting=cfg["model_cfg"]["model_setting"],
    #     )
    # if not cfg["model_cfg"]["model_setting"].get("scratch", False):
    #     unigs.pointnet.load_state_dict(torch.load(cfg.resume_from, map_location="cpu"))

    try:
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
    except:
        unigs = torch.load(cfg.resume_from, map_location="cpu")
        if not isinstance(unigs, UniGS): unigs = unigs.module

    return unigs

def log_epoch(text_accuracy, image_accuracy, total_num,
              results_tb, collect_data_list, collect_data):
    results_tb_rows = [
        ["Text", str(text_accuracy.item())+"/"+str(total_num)],
        ["accuracy", round((text_accuracy/total_num).item(),4)],
        ["Image ", str(image_accuracy.item())+"/"+str(total_num)],
        ["accuracy", round((image_accuracy/total_num).item(),4)],
    ]
    for key in collect_data_list:
        results_tb_rows[0].append( str(collect_data[key][0]) + "/" + str(collect_data[key][2]) )
        results_tb_rows[1].append( str(round(collect_data[key][0]/(collect_data[key][2]+1e-8),4)) )
        results_tb_rows[2].append( str(collect_data[key][1]) + "/" + str(collect_data[key][2]) )
        results_tb_rows[3].append( str(round(collect_data[key][1]/(collect_data[key][2]+1e-8),4)) )
    results_tb.add_row(results_tb_rows[0])
    results_tb.add_row(results_tb_rows[1], divider=True)
    results_tb.add_row(results_tb_rows[2])
    results_tb.add_row(results_tb_rows[3])
    print(results_tb, flush=True)

@torch.no_grad()
def test_retrive(model, test_dataloader, device, k, local_rank_flag=True, first_epoch_flag=True, get_image=False):
    # text
    text_list, text_item_dict = test_dataloader.dataset.gather_retrive_prompts()
    text_features, text_embed_chunk = [], 24
    for i in range( math.ceil(len(text_list)/text_embed_chunk) ):
        text_embeds = clip.tokenize(text_list[i*text_embed_chunk : (i+1)*text_embed_chunk]).to(device)
        text_feature = model.encode_text(text_embeds).to(torch.float32).cpu().detach()
        text_features.append(text_feature)
        del text_embeds, text_feature
    text_features = torch.cat(text_features)
    text_features = nn.functional.normalize(text_features, dim=1, p=2)

    # img
    if get_image:
        img_list = []
        for meta_data in test_dataloader:
            img_list.append(meta_data["image"])
        img_features, img_embed_chunk = [], 24
        for i in range( math.ceil(len(img_list)/img_embed_chunk) ):
            img_embeds = torch.cat(img_list[i*img_embed_chunk : (i+1)*img_embed_chunk]).to(device)
            img_feature = model.encode_image(img_embeds).to(torch.float32).cpu().detach()
            img_features.append(img_feature)
            del img_embeds, img_feature
        img_features = torch.cat(img_features)
        img_features = nn.functional.normalize(img_features, dim=1, p=2)


    model.eval()
    logit_scale = model.logit_scale.exp()
    total_num = 0

    accuracy_dict = {}
    for k_single in k: accuracy_dict[k_single] = [0, 0]
    
    if local_rank_flag and first_epoch_flag: test_dataloader_iterator = tqdm(test_dataloader, desc="Evaling all")
    else: test_dataloader_iterator = test_dataloader

    for meta_data in test_dataloader_iterator:
        guassian = meta_data["gaussian"]
        item_paths = meta_data["item_path"]
        batch = guassian.shape[0]
        image_gt_label = torch.arange(batch, device=device)

        text_gt_label, text_mask = [], []
        for item_path in item_paths:
            text_gt_label.append(text_item_dict[item_path]["count"])
            text_mask.append(text_item_dict[item_path]["text_mask"])
        text_gt_label = torch.Tensor(text_gt_label).to(device)
        text_mask = torch.Tensor(np.array(text_mask)).bool().to(device)

        gaussian_features, _ = model.encode_gaussian(guassian.to(device)) # [2, 512]

        # normalized features
        gaussian_features = nn.functional.normalize(gaussian_features, dim=1, p=2)

        text_feature_gpu = text_features.to(device).t()
        logits_per_text = logit_scale * gaussian_features @ text_feature_gpu
        logits_per_text[text_mask] = 0

        for k_single in k:

            _, topk_text_label = torch.topk(logits_per_text, k_single, dim=-1)
            accuracy_dict[k_single][0] += torch.sum( (topk_text_label==text_gt_label.unsqueeze(1)).any(dim=1) )

            if get_image:
                img_feature_gpu = img_features.to(device).t()
                logits_per_image = logit_scale * gaussian_features @ img_feature_gpu

                _, topk_img_label = torch.topk(logits_per_image, k_single if k_single < batch else batch, dim=-1)
                accuracy_dict[k_single][1] += torch.sum( (topk_img_label==image_gt_label.unsqueeze(1)).any(dim=1) )

        total_num += batch

    if local_rank_flag:
        for k_single in k:
            print("[Testing] Top %d Text-3D accuracy: %.4f"%(k_single, accuracy_dict[k_single][0]/total_num))
            if get_image: print("[Testing] Top %d Image-3D accuracy: %.4f"%(k_single, accuracy_dict[k_single][1]/total_num))
    
    return total_num, accuracy_dict[k[0]][0]

@torch.no_grad()
def test_classification(model, test_dataloader, device, class_to_num,
                        test_text, data_type_list, collect_data_list, k=[1]):
    model.eval()

    collect_data = {key:[0,0,0] for key in collect_data_list} # text image all

    total_num = 0

    accuracy_dict, collect_data_dict = {}, {}
    for k_single in k:
        accuracy_dict[k_single] = [0, 0]
        collect_data_dict[k_single] = collect_data.copy()

    test_dataloader_iterator = tqdm(test_dataloader, desc="Evaling all")
    for meta_data in test_dataloader_iterator:
        guassian = meta_data["gaussian"]
        batch = guassian.shape[0]

        image_gt_label = torch.arange(batch, device="cuda")
        text_gt_label = meta_data["label_count"].to(device)

        logits_per_image, logits_per_text = model.predict(
            test_text, meta_data["image"].to(device), meta_data["gaussian"].to(device), use_mask=(test_text is None), return_probability=True
        )
    
        total_num += batch

        for k_single in k:
            _, topk_text_label = torch.topk(logits_per_text, k_single, dim=-1)
            topk_text_retrieve_mask = (topk_text_label==text_gt_label.unsqueeze(1)).any(dim=1)
            text_label = text_gt_label * topk_text_retrieve_mask
            accuracy_dict[k_single][0] += torch.sum( topk_text_retrieve_mask )

            _, topk_img_label = torch.topk(logits_per_image, k_single if k_single < batch else batch, dim=-1)
            topk_image_retrieve_mask = (topk_img_label==image_gt_label.unsqueeze(1)).any(dim=1)
            image_label = image_gt_label * topk_image_retrieve_mask
            accuracy_dict[k_single][1] += torch.sum( topk_image_retrieve_mask )

            for j in range(len(data_type_list)):
                data_type = data_type_list[j].lower()
                this_label = int(class_to_num[data_type])
                if data_type in collect_data_list:
                    collect_data_dict[k_single][data_type][0] += torch.sum((text_gt_label==this_label) * (text_gt_label==text_label)).item()
                    collect_data_dict[k_single][data_type][2] += torch.sum(text_gt_label==this_label).item()
                    collect_data_dict[k_single][data_type][1] += torch.sum((text_gt_label==this_label) * (image_gt_label==image_label)).item()

    for k_single in k:
        print("Top %d"%(k_single))
        log_epoch(accuracy_dict[k_single][0], accuracy_dict[k_single][1], total_num, PrettyTable(["", "3D accuracy", *collect_data_list]), collect_data_dict[k_single], collect_data)
        print("\n")
    
    return total_num

@torch.no_grad()
def test_abo_classification(model, test_dataloader, device, k):
    ABO_type_list = ["bed","bench","cabinet","chair","desk","dresser","furniture","home","lamp","mirror","ottoman",
                    "pillow","planter","rug","shelf","sofa","stool","storage","table","vase","wall_art"]
    # ABO_collect_list = ["bed","bench","cabinet","chair","desk","dresser","furniture","home","lamp","mirror","ottoman",
    #                 "pillow","planter","rug","shelf","sofa","stool","storage","table","vase","wall_art"]
    ABO_collect_list = ["cabinet", "chair", "desk", "dresser", "lamp", "mirror", "pillow", "rug", "shelf", "sofa", "table", "vase", "wall_art"]
    data_type_list, collect_data_list = ABO_type_list, ABO_collect_list

    pc_prefix, test_text = "point cloud of ", []
    for i in range(len(data_type_list)):
        test_text.append(pc_prefix+data_type_list[i])

    class_to_num = {}
    with open("/path/to/your/gaussian-splatting/unigs/ABO/category.json", "r") as file:
        class_to_num = json.load(file)

    # file_paths = [] 
    # selected_class = ["cabinet", "chair", "desk", "dresser", "lamp", "mirror", "pillow", "rug", "shelf", "sofa", "table", "vase", "wall_art"]
    # for file_path in test_dataloader.dataset.file_paths:
    #     this_class = file_path.split("/")[-2]
    #     if this_class in selected_class: file_paths.append(file_path)
    # test_dataloader.dataset.file_paths = file_paths

    return test_classification(model, test_dataloader, device, class_to_num, test_text, data_type_list, collect_data_list, k)

@torch.no_grad()
def test_sunrgbd_classification(model, test_dataloader, device, k):
    SUNRGBD_type_list = ["wall","floor","cabinet","bed","chair","sofa","table","door",
                    "window","bookshelf","picture","counter","blinds","desks","shelves",
                    "curtain","dresser","pillow","mirror","floor-mat","clothes","ceiling",
                    "books","refrigerator","television","paper","towel","shower-curtain",
                    "box","whiteboard","person","nightstand","toilet","sink","lamp","bathtub","bag"]
    SUNRGBD_collect_list = ["bed","bookshelf","chair","desks","sofa","table","toilet","bathtub","dresser","nightstand"]
    data_type_list, collect_data_list = SUNRGBD_type_list, SUNRGBD_collect_list

    pc_prefix, test_text = "point cloud of ", []
    for i in range(len(data_type_list)):
        test_text.append(pc_prefix+data_type_list[i])

    class_to_num = {}
    with open("/path/to/your/SUNRGBD/label_inverse.json", "r") as file:
        class_to_num = json.load(file)
    for key in list(class_to_num.keys()):
        if key != key.lower(): class_to_num[key.lower()] = class_to_num[key]

    return test_classification(model, test_dataloader, device, class_to_num, test_text, data_type_list, collect_data_list, k)

@torch.no_grad()
def test_objaverse_classification(model, test_dataloader, device, k, update_label):
    Objaverse_type_list = []
    lvis_annotation_path = "/path/to/your/objarverse/openshape/meta_data/split/lvis.json"
    with open(lvis_annotation_path, "r") as file:
        lvis_content = json.load(file)
    for lvis_item in lvis_content:
        if lvis_item["category"] not in Objaverse_type_list: Objaverse_type_list.append(lvis_item["category"])

    Objaverse_collect_list = ["mug","owl","mushroom","fireplug","banana","ring","doughnut","armor","sword","control","cone",
                        "gravestone","chandelier","snowman","shield","antenna","seashell","chair"]
    data_type_list, collect_data_list = Objaverse_type_list, Objaverse_collect_list

    class_to_num = {}
    with open("/path/to/your/objarverse/label.json", "r") as file:
        class_to_num = json.load(file)

    if update_label:    
        # label 50
        with open("/path/to/your/objarverse/label_50.json", "r") as file:
            class_to_num = json.load(file)
        data_type_list = list(class_to_num.keys())

    pc_prefix, test_text = "point cloud of ", []
    for i in range(len(data_type_list)):
        test_text.append(pc_prefix+data_type_list[i])

    return test_classification(model, test_dataloader, device, class_to_num, test_text, data_type_list, collect_data_list, k)

@torch.no_grad()
def test_mvimgnet_classification(model, test_dataloader, device, k, update_label):
    Mvimgnet_annotation_path = "/path/to/your/MVimgnet/scripts/mvimgnet_category.txt"
    Mvimgnet_type_list = []
    with open(Mvimgnet_annotation_path, "r") as file:
        for line in file.readlines():
            line = line.replace("\n","").split(",")
            Mvimgnet_type_list.append(line[1].lower())
    Mvimgnet_collect_list = ['bottle', 'conch', 'tangerine', 'okra', 'guava', 'bulb', 'bag', 'glove', 
                              'accessory', 'garlic', 'lipstick', 'telephone', 'watch', 'lock', 'bowl', 'toothpaste']
    data_type_list, collect_data_list = Mvimgnet_type_list, Mvimgnet_collect_list

    class_to_num = {}
    with open("/path/to/your/MVimgnet/scripts/mvimgnet_category.txt", "r") as file:
        for line in file.readlines():
            line = line.replace("\n","").split(",")
            class_to_num[line[1]] = line[0]
    
    if update_label:    
        # label 50
        with open("/path/to/your/MVimgnet/scripts/label_95.json", "r") as file:
            class_to_num = json.load(file)
        data_type_list = list(class_to_num.keys())

    pc_prefix, test_text = "point cloud of ", []
    for i in range(len(data_type_list)):
        test_text.append(pc_prefix+data_type_list[i])

    return test_classification(model, test_dataloader, device, class_to_num, test_text, data_type_list, collect_data_list, k)


@torch.no_grad()
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cfg.seed = args.seed
    device = torch.device("cuda:"+str(args.gpu_id))

    model = load_checkpoint(cfg).to(device)

    if args.data_split != "": cfg.data_cfg.dataset.data_split = args.data_split
    if args.dataset != "": cfg.data_cfg.dataset.type = args.dataset
    if args.data_dir != "": cfg.data_cfg.dataset.data_dir = args.data_dir

    test_dataset = build_dataset(cfg.data_cfg.dataset, args.update_label)

    if args.batch_size < 0:
        batch_size = cfg["data_cfg"].get("test_batch_size", cfg["data_cfg"]["batch_size"])
    else: batch_size = args.batch_size

    num_works = cfg["data_cfg"]["nw"]

    pin_memory = cfg["data_cfg"]["pin_memory"]

    if args.task == "":
        task = cfg["test_cfg"]["task"]
    else: task = args.task

    if args.test_dataset == "":
        dataset = cfg["test_cfg"]["dataset"]
    else: dataset = args.test_dataset

    args.k = [int(k) for k in args.k.split(",")]

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_works, pin_memory=pin_memory)
    
    if task == "retrive":
        test_retrive(model, test_dataloader, device, k=args.k)
    elif task == "classification":
        if dataset == "abo":
            test_abo_classification(model, test_dataloader, device, args.k)
        elif dataset == "sunrgbd":
            test_sunrgbd_classification(model, test_dataloader, device, args.k)
        elif dataset == "objaverse":
            test_objaverse_classification(model, test_dataloader, device, args.k, args.update_label)
        elif dataset == "mvimgnet":
            test_mvimgnet_classification(model, test_dataloader, device, args.k, args.update_label)
        else:
            raise




if __name__ == '__main__':
    main()