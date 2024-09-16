import os
import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import time
from tqdm import tqdm
import random
import json

def load_ply(path, offset=[0,0,0], max_sh_degree=0):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"])+offset[0],
                    np.asarray(plydata.elements[0]["y"])+offset[1],
                    np.asarray(plydata.elements[0]["z"])+offset[2]),  axis=1)
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

data_root = "//path/to/your/MVimgnet/mvi"
raw_data_root = "/path/to/your/gaussian-splatting/output/mvimgnet"
save_data_root = "/path/to/your/gaussian-splatting/clip3/mvimgnet" 
class_annotation_path = "//path/to/your/MVimgnet/scripts/mvimgnet_category.txt"

class_to_num = {}
with open(class_annotation_path, "r") as file:
    for line in file.readlines():
        line = line.replace("\n","").split(",")
        class_to_num[line[1].lower()] = line[0]

file_paths = {
    "original":{
        "train":[], 
        "test":[],
        "val":[]
    },
    "all":{},
}
for data_type in tqdm(os.listdir(raw_data_root)):
    # if not data_type=="bookshelf": continue

    data_type_lower = data_type.lower()
    data_type_root = os.path.join(raw_data_root, data_type)
    save_data_type_root = os.path.join(save_data_root, "objects", data_type_lower)

    file_paths["all"][data_type_lower] = []
    if len(os.listdir(data_type_root)) > 0: os.makedirs(save_data_type_root, exist_ok=True)
    else: continue

    item_name_count = 0
    for item_id in os.listdir(data_type_root):
        save_item_path = os.path.join(save_data_root, "objects", data_type_lower, item_id+".pkl")
        if os.path.exists(save_item_path): 
            continue
        img_root = os.path.join(data_root, class_to_num[data_type], item_id, "images")

        item_ply_path = os.path.join(data_type_root, item_id, "point_cloud", "iteration_500", "point_cloud.ply")
        if not os.path.exists(item_ply_path): continue
        file_paths["all"][data_type_lower].append(os.path.join(save_data_root, "objects", data_type_lower, pkl_name))
        ply_feature = load_ply(item_ply_path)

        save_item_data = {
            "name":item_id,
            "label": data_type_lower,
            "label_count": class_to_num[data_type_lower],
            "dataset": "mvimgnet",
            "item_path":save_item_path,
            "img":img_root,
            "3dgs":ply_feature.data.numpy(),
        }
        torch.save(save_item_data, save_item_path)

only_train = True
if not os.path.exists(os.path.join(save_data_root, "file_paths.json")):
    rate = 0.8
    for data_type in file_paths["all"].keys():
        item_id_list = file_paths["all"][data_type]
        random.shuffle(item_id_list)
        train_split_ptr = int(len(item_id_list)*0.8)
        test_split_ptr = int(len(item_id_list)*0.1)
        if only_train:
            file_paths["original"]["train"].extend(item_id_list[:train_split_ptr+test_split_ptr])
            file_paths["original"]["test"].extend(item_id_list[train_split_ptr+test_split_ptr:])
        else:
            file_paths["original"]["train"].extend(item_id_list[:train_split_ptr])
            file_paths["original"]["test"].extend(item_id_list[train_split_ptr:train_split_ptr+test_split_ptr])
            file_paths["original"]["val"].extend(item_id_list[train_split_ptr+test_split_ptr:])

    with open(os.path.join(save_data_root, "file_paths.json"), "w") as file:
        json.dump(file_paths, file, indent=4)