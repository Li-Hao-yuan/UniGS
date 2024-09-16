import os
import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import time
from tqdm import tqdm
import random
import json

original_type_list = ['storage rack', 'bat shelter', 'sectional sofa lounge set', 'clothes rack', 'furniture', 'furniture cover', 'wall art', 'tablet holder stand', 
                      'wicker planter', 'professional healthcare chair', 'flat screen display mount', 'cabinet furniture', 'light fixture', 'garden utility wagon', 
                      'wireless speaker', 'computer', 'birdhouse', 'fitness bench', 'canopy', 'mattress', 'carrying case', 'electronic device cover', 'basket', 
                      'electronic cable', 'bean bag chair', 'cable', 'artificial plant', 'file folder', 'jar', 'rack', 'laptop desk stand', 'mouse pad', 
                      'laptop stand arm mount tray', 'office_products', 'ottoman bench', 'fry pan', 'step stool', 'speaker bracket', 'jewelry storage', 'stool', 
                      'air conditioner', 'home', 'ottoman', 'bottle rack', 'laundry hamper', 'bench', 'table', 'mat', 'monitor stand', 'audio cable', 'computer holder', 
                      'writing board', 'electronic adapter', 'placemat', 'home furniture and decor', 'headboard', 'bed frame', 'figurine', 'drink coaster', 'utility cart wagon', 
                      'hook', 'shelf', 'exercise mat', 'fan', 'chair', 'vase', 'ottoman chair', 'sofa bench', 'night stand', 'mirror', 'pump', 'auto accessory', 
                      'outdoor garden utility wagon', 'electric fan', 'curtain', 'candle holder', 'box', 'pillow', 'light', 'wireless cable', 'sofa', 'storage bench', 
                      'tableware bowl', 'keyboard and piano stand', 'slim trash can', 'door organizer', 'photo frame', 'nightstand table', 'battery', 'clock', 'janitorial supply', 
                      'planter', 'drinking cup', 'heater cover', 'ottoman sofa', 'lamp', 'pillow cover', 'storage', 'nightstand', 'bed', 'rug', 'paper product', 'dresser', 
                      'outdoor pyramid patio heater', 'cabinet', 'picture frame', 'consumer electronics', 'desk', 'screen display mount', 'multiport hub', 'laptop and projector stand', 
                      'outdoor patio firepit table', 'sporting goods', 'rowing machine', 'tablets portable stand', 'mount', 'dehumidifier', 'tv stand', 'ladder']


transfer_type_mapping1 = {
    "rack":["storage rack", 'clothes rack', 'bottle rack', 'rack'],
    "furniture":['furniture', 'home furniture and decor', 'furniture cover'],
    "cabinet":['cabinet', 'cabinet furniture'],
    "lamp":['lamp', 'light', 'light fixture'], 
    "ottoman":['ottoman bench', 'ottoman', 'ottoman chair', 'ottoman sofa'],
    "sofa":['sofa bench', 'sofa', 'sectional sofa lounge set'],
    "cable":['cable', 'audio cable', 'wireless cable', 'electronic cable', 'electronic adapter'],
    "planter":['planter', 'artificial plant'],
    'wall art':['wall art', 'photo frame', 'picture frame'],
    'shelf':['shelf'],
    'chair':['chair', 'professional healthcare chair', 'bean bag chair'],
    'vase':['vase'],
    'mattress':['mattress'],
    'stool':['stool', 'step stool'],
    'outdoor living':['outdoor pyramid patio heater', 'outdoor patio firepit table', 'wicker planter', 'garden utility wagon', 'outdoor garden utility wagon', 'utility cart wagon',
                      'birdhouse', "bat shelter", 'heater cover'],
    'bench':['bench', 'storage bench', 'fitness bench'],
    'pillow':['pillow', 'pillow cover'],
    'bed':['bed', 'bed frame', 'headboard'],
    'mirror':['mirror'],
    'computer':['computer', 'computer holder'],
    "ectronic device stand":['monitor stand', "laptop desk stand", "flat screen display mount", 'laptop stand arm mount tray', 'screen display mount', 'mount', 'tv stand',
                             "keyboard and piano stand", 'tablets portable stand', 'laptop and projector stand', 'tablet holder stand'],
    'fan':['fan', 'electric fan'],
    'nightstand':['nightstand', 'night stand'],
    'table':['table', 'nightstand table'],
    "box":["tableware bowl", 'box'],
    "wireless speaker":["wireless speaker"],
    'canopy':['canopy'],
    'basket':["basket", 'laundry hamper'],
    'mouse pad':['mouse pad'],
    'office products':["office_products"],
    'storage':['storage', 'jewelry storage', 'hook'],
    'air conditioner':["air conditioner"],
    'home':['home', 'candle holder', 'slim trash can', 'dehumidifier', 'ladder'],
    'speaker bracket':['speaker bracket'],
    'mat':['mat', 'placemat', 'exercise mat', 'drink coaster'],
    'writing board':['writing board'],
    "curtain":['curtain'],
    'clock':['clock'],
    'rug':['rug'],
    'paper product':['paper product'],
    'dresser':['dresser'],
    'desk':['desk'],
    'multiport hub':['multiport hub'],
    'sporting goods':['sporting goods'],
}

mapping_type_dict = {}
for key in transfer_type_mapping1.keys():
    for item_type in transfer_type_mapping1[key]:
        mapping_type_dict[item_type]=key

mapping_type_count = {'computer': 3, 'curtain': 3, 'wireless speaker': 4, 'paper product': 4, 'mouse pad': 5, 'writing board': 5, 'office products': 6, 'speaker bracket': 6, 'box': 8, 
'basket': 11, 'clock': 11, 'outdoor living': 12, 'air conditioner': 12, 'cable': 13, 'multiport hub': 16, 'mattress': 17, 'canopy': 20, 'nightstand': 24, 'sporting goods': 28, 
'fan': 29, 'mat': 30, 'rack': 32, 'ectronic device stand': 43, 'home': 50, 'storage': 53, 'dresser': 59, 'vase': 60, 'bench': 61, 'shelf': 97, 'furniture': 141, 'mirror': 150, 
'desk': 153, 'cabinet': 165, 'planter': 181, 'pillow': 279, 'ottoman': 359, 'stool': 375, 'bed': 438, 'lamp': 564, 'table': 610, 'wall art': 688, 'rug': 866, 'sofa': 953, 'chair': 1276}

############

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

    extra_f_names = []

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

item_thresh = 50
img_root = "//path/to/your/ABO/render_images_256"
raw_data_root = "/path/to/your/gaussian-splatting/output/ABO"
save_data_root = "/path/to/your/gaussian-splatting/clip3/ABO" 

left_mapping_type_list, left_mapping_dict = [], {}
for key in mapping_type_count:
    if mapping_type_count[key]>=item_thresh: left_mapping_type_list.append(key)
left_mapping_type_list.sort()
for i,key in enumerate(left_mapping_type_list):
    left_mapping_dict[key.replace(" ","_")] = i
with open(os.path.join(save_data_root, "category.json"), "w") as file:
    json.dump(left_mapping_dict, file, indent=4)

type_json_path = "//path/to/your/ABO/ABO_prompt_type.json"
GPT_prompt_json_path = "//path/to/your/ABO/ABO_GPT_prompt_all.json"
with open(type_json_path, "r") as file:
    type_json = json.load(file)
with open(GPT_prompt_json_path, "r") as file:
    gpt_prompt_json = json.load(file)

file_paths = {
    "original":{
        "train":[], 
        "test":[],
        "val":[]
    },
    "all":{},
}

for item_id in tqdm(os.listdir(raw_data_root)):

    if item_id in ["B07847Y5BG"]: continue

    product_type = type_json[item_id]["product_type"].lower()
    additional_type = type_json[item_id]["additional_type"].lower()

    if additional_type not in mapping_type_dict.keys(): continue
    additional_type = mapping_type_dict[additional_type]

    if mapping_type_count[additional_type]<item_thresh: continue

    product_type = product_type.replace(" ", "_")
    additional_type = additional_type.replace(" ", "_")

    save_data_type_root = os.path.join(save_data_root, "objects", additional_type) 
    os.makedirs(save_data_type_root, exist_ok=True)

    if additional_type not in file_paths["all"].keys():
        file_paths["all"][additional_type] = []

    save_item_path = os.path.join(save_data_root, "objects", additional_type, item_id+".pkl")
    # if os.path.exists(save_item_path): continue
    this_img_root = os.path.join(img_root, product_type, item_id, "up")

    item_ply_path = os.path.join(raw_data_root, item_id, "point_cloud", "iteration_2000", "point_cloud.ply")
    if not os.path.exists(item_ply_path): continue
    file_paths["all"][additional_type].append(save_item_path)

    ply_feature = load_ply(item_ply_path)

    save_item_data = {
        "name":item_id,
        "label": additional_type,
        "label_count":left_mapping_dict[additional_type],
        "dataset": "abo",
        "machine_prompt": gpt_prompt_json[item_id],
        "item_path":save_item_path,
        "img":this_img_root,
        "3dgs":ply_feature.data.numpy(),
        
        "product_type": product_type,
    }
    torch.save(save_item_data, save_item_path)
    # exit()

only_train = False
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