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

data_root = "//path/to/your/objarverse/views_release"
raw_data_root = "/path/to/your/gaussian-splatting/output/objaverse"
save_data_root = "/path/to/your/gaussian-splatting/clip3/objaverse_all" 
os.makedirs(save_data_root, exist_ok=True)
os.makedirs(os.path.join(save_data_root, "objects"), exist_ok=True)

# select datasets
select_item_list = []
select_json_path = "/path/to/your/gaussian-splatting/scripts/Objaverse/split/lvis.json"
with open(select_json_path, "r") as file:
    select_item_list = json.load(file)

## category 
lvis_annotation_dict, categoy_list = {}, []
lvis_annotation_path = "//path/to/your/objarverse/openshape/meta_data/split/lvis.json"
with open(lvis_annotation_path, "r") as file:
    lvis_content = json.load(file)
for lvis_item in lvis_content:
    lvis_annotation_dict[lvis_item["uid"]] = lvis_item["category"]
    if lvis_item["category"] not in categoy_list: categoy_list.append(lvis_item["category"])

## load label json
label_json_path = "//path/to/your/objarverse/label.json"
if os.path.exists(label_json_path):
    with open(label_json_path, "r") as file:
        category_to_num = json.load(file)

else:
    categoy_list.sort()
    category_to_num = {}
    for i,category in enumerate(categoy_list):
        category_to_num[category.lower()] = i
    
    with open(label_json_path, "w") as file:
        json.dump(category_to_num, file, indent=4)
# exit()

prompt_json_path = "/path/to/your/gaussian-splatting/data/objaverse/file_paths.json"
with open(prompt_json_path, "r") as file:
    prompt_json = json.load(file)

exclude_item_list = []
exclude_json_path = "/path/to/your/gaussian-splatting/scripts/Objaverse/split/exclude.json"
if os.path.exists(exclude_json_path):
    with open(exclude_json_path, "r") as file:
        exclude_item_list = json.load(file)

file_paths = {
    "original":{
        "train":[], 
        "test":[],
        "val":[]
    },
    "all":[],
}


human_caption_item_list, only_machine_caption_item_list = [], []
for item_id in tqdm(os.listdir(raw_data_root)):
# for item_id in ["c130a2b8c219469cb1e82ee190be67ee"]:
    if item_id not in select_item_list: continue
    if item_id in exclude_item_list: continue
    # if not item_id == "95a6c88077c4431d8eab7aee8cb28a00": continue

    item_root = os.path.join(raw_data_root, item_id)
    if not os.path.isdir(item_root) or "." in item_id: continue
    pkl_name = item_id+".pkl"

    save_item_path = os.path.join(save_data_root, "objects", pkl_name)
    file_paths["all"].append(save_item_path)

    save_item_data = {}
    machine_prompt:str = prompt_json[item_id]["prompt_machine"]
    human_prompt:list = prompt_json[item_id]["prompt_human"]

    assert len(machine_prompt)>0 or len(human_prompt[0])>0 or item_id in lvis_annotation_dict.keys(), print(item_id)

    if len(machine_prompt)>0: save_item_data["machine_prompt"] = machine_prompt
    if len(human_prompt[0])>0: save_item_data["human_prompt"] = human_prompt

    if os.path.exists(save_item_path): 
        if len(human_prompt[0])>0: 
            human_caption_item_list.append(save_item_path)
        else: only_machine_caption_item_list.append(save_item_path)
        continue
    img_root = prompt_json[item_id]["img_path"]
    glb_path = prompt_json[item_id]["pts_path"]

    item_ply_path = os.path.join(item_root, "point_cloud", "iteration_3000", "point_cloud.ply")

    exist_flag, item_ply_path = False, ""
    for iteration in [1500, 2000, 3000]:
        iteration_ply_path = os.path.join(item_root, "point_cloud", "iteration_"+str(iteration), "point_cloud.ply")
        exist_flag = exist_flag or os.path.exists(iteration_ply_path)
        if exist_flag: 
            item_ply_path = iteration_ply_path
            break

    if not exist_flag: continue

    try:
        ply_feature = load_ply(item_ply_path)
    except:
        print("Error: ", item_ply_path)
        continue

    if len(human_prompt[0])>0: 
        human_caption_item_list.append(save_item_path)
    else: only_machine_caption_item_list.append(save_item_path)

    if item_id in lvis_annotation_dict.keys():
        label = lvis_annotation_dict[item_id]
        label_count = category_to_num[label]
    else: label, label_count = "", 0

    save_item_data["name"] = item_id
    save_item_data["item_path"] = save_item_path
    save_item_data["label_count"] = label_count
    save_item_data["dataset"] = "objaverse"
    save_item_data["label"] = label
    save_item_data["img"] = img_root
    save_item_data["3dgs"] = ply_feature.data.numpy()

    save_item_data["glb_path"] = glb_path
    
    torch.save(save_item_data, save_item_path)


random_sample = False
only_train = True
if not os.path.exists(os.path.join(save_data_root, "file_paths.json")):
    if random_sample:
        rate = 0.8
        item_id_list = file_paths["all"].copy()
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
    else:
        random.shuffle(human_caption_item_list)
        random.shuffle(only_machine_caption_item_list)
        test_split_ptr, val_split_ptr = 1000, 1000
        if only_train:
            file_paths["original"]["test"].extend(human_caption_item_list[:test_split_ptr])

            file_paths["original"]["train"].extend(human_caption_item_list[test_split_ptr:])
            file_paths["original"]["train"].extend(only_machine_caption_item_list)
        else:
            file_paths["original"]["test"].extend(human_caption_item_list[:test_split_ptr])
            file_paths["original"]["val"].extend(human_caption_item_list[test_split_ptr:test_split_ptr+val_split_ptr])

            file_paths["original"]["train"].extend(human_caption_item_list[test_split_ptr+val_split_ptr:])
            file_paths["original"]["train"].extend(only_machine_caption_item_list)

    with open(os.path.join(save_data_root, "file_paths.json"), "w") as file:
        json.dump(file_paths, file, indent=4)
