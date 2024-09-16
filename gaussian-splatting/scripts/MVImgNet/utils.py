import os
import json
from tqdm import tqdm
import torch

def transfer_sunrgbd():
    def add_pkl_attribute(root, dataset_name, class_to_num, prefix=""):
        for pkl_name in tqdm(os.listdir(root),desc=prefix):
            pkl_path = os.path.join(root, pkl_name)
            if pkl_path.endswith("json"): continue
            if os.path.isdir(pkl_path): 
                add_pkl_attribute(pkl_path, dataset_name, class_to_num, pkl_name)
                print()
            else:
                pkl_data = torch.load(pkl_path)
                if "dataset" not in pkl_data.keys() or pkl_data["dataset"] != dataset_name: 
                    pkl_data["dataset"] = dataset_name
                if "label" not in pkl_data.keys() or pkl_data["label"] != prefix:
                    pkl_data["label"] = prefix

                if dataset_name == "sunrgbd":
                    if prefix not in transfer_dict.keys():
                        pkl_data["label_count"] = -1
                    else:
                        pkl_data["label_count"] = class_to_num[transfer_dict[prefix]]

                torch.save(pkl_data, pkl_path)

    data_mapping = {
            "wall":[],
            "floor":[],
            "cabinet":["cabinet"],
            "bed":["bed","bunk bed","baby bed"],
            "chair":["chair","child chair","saucer chair","stack of chairs","high chair","lounge chair","baby chair","bench"],
            "sofa":["sofa","sofa bed","sofa chair"],
            "table":["side table","table","coffee table","dining table","end table","foosball table","ping pong table"],
            "door":["door"],
            "window":["window"],
            "bookShelf":["bookshelf"],
            "picture":["picture",],
            "counter":["counter","cupboard","closet",],
            "blinds":["blinds",],
            "desks":["desk"],
            "shelves":["shelves", "shelf",],
            "curtain":["window shade","curtain",],
            "dresser":["dresser",],
            "pillow":["pillow"],
            "mirror":["dresser mirror","mirror"],
            "floor-mat":["mat","bathmat","carpet"],
            "clothes":["cloth","coat","jacket"],
            "ceiling":[],
            "books":["notebook","book","books","magazine"],
            "refrigerator":["mini refrigerator"],
            "television":["tv"],
            "paper":["newspaper","paper","paper ream","paper towel","paper towel dispenser","tissuebox","toilet paper","toiletpaper"],
            "towel":["towel"],
            "shower-curtain":["shower curtain"],
            "box":["box","pizza box"],
            "whiteboard":["whiteboard","bulletin board","blackboard","chalkboard","bulletin","board"],
            "person":["person"],
            "nightStand":["nightstand"],
            "toilet":["toilet"],
            "sink":["sink"],
            "lamp":["lamp","light","lighting fixture"],
            "bathtub":["bathtub"],
            "bag":["paper bag","plastic bag","bag","bags"]
        }
    transfer_dict = {}
    for key in data_mapping.keys():
        for value in data_mapping[key]:
            transfer_dict[value] = key

    root = "/path/to/your/gaussian-splatting/clip3/sunrgbd_all/objects"
    with open("//path/to/your/SUNRGBD/label_inverse.json", "r") as file:
        class_to_num = json.load(file)

    dataset_name="sunrgbd"
    add_pkl_attribute(root, dataset_name, class_to_num)

def transfer_abo():
    def add_pkl_attribute(root, dataset_name, class_to_num, prefix=""):
        for pkl_name in tqdm(os.listdir(root),desc=prefix):
            pkl_path = os.path.join(root, pkl_name)
            if pkl_path.endswith("json"): continue
            if os.path.isdir(pkl_path): add_pkl_attribute(pkl_path, dataset_name, class_to_num, pkl_name)
            else:
                pkl_data = torch.load(pkl_path)
                if "dataset" not in pkl_data.keys() or pkl_data["dataset"] != dataset_name: 
                    pkl_data["dataset"] = dataset_name
                if "label" not in pkl_data.keys() or pkl_data["label"] != prefix:
                    pkl_data["label"] = prefix
                if "label_count" not in pkl_data.keys() or pkl_data["label_count"] != class_to_num[prefix]:
                    pkl_data["label_count"] = int(class_to_num[prefix])

                torch.save(pkl_data, pkl_path)

    root = "/path/to/your/gaussian-splatting/clip3/ABO/objects"
    with open("/path/to/your/gaussian-splatting/clip3/ABO/category.json", "r") as file:
        class_to_num = json.load(file)

    dataset_name="abo"
    add_pkl_attribute(root, dataset_name, class_to_num)

def transfer_mvimgnet():
    def add_pkl_attribute(root, dataset_name, class_to_num, prefix=""):

        for pkl_name in tqdm(os.listdir(root),desc=prefix):
            pkl_path = os.path.join(root, pkl_name)
            if pkl_path.endswith("json"): continue
            if os.path.isdir(pkl_path): add_pkl_attribute(pkl_path, dataset_name, class_to_num, pkl_name)
            else:
                pkl_data = torch.load(pkl_path)
                if "dataset" not in pkl_data.keys() or pkl_data["dataset"] != dataset_name: 
                    pkl_data["dataset"] = dataset_name
                if "label" not in pkl_data.keys() or pkl_data["label"] != prefix:
                    pkl_data["label"] = prefix
                if "label_count" not in pkl_data.keys() or pkl_data["label_count"] != class_to_num[prefix]:
                    pkl_data["label_count"] = int(class_to_num[prefix])
                pkl_data['img'] = pkl_data['img'].replace("/mvi_00/","/mvi/")

                torch.save(pkl_data, pkl_path)

    root = "/path/to/your/gaussian-splatting/clip3/mvimgnet_500/objects"
    class_annotation_path = "//path/to/your/MVimgnet/scripts/mvimgnet_category.txt"
    class_to_num = {}
    with open(class_annotation_path, "r") as file:
        for line in file.readlines():
            line = line.replace("\n","").split(",")
            class_to_num[line[1].lower()] = line[0]

    dataset_name="mvimgnet"
    add_pkl_attribute(root, dataset_name, class_to_num)

def transfer_objaverse():
    def add_pkl_attribute(root, dataset_name, class_to_num, lvis_item_dict, prefix=""):

        for pkl_name in tqdm(os.listdir(root),desc=prefix):
            pkl_path = os.path.join(root, pkl_name)
            if pkl_path.endswith("json"): continue
            if os.path.isdir(pkl_path): add_pkl_attribute(pkl_path, dataset_name, class_to_num, pkl_name)
            else:
                pkl_data = torch.load(pkl_path)
                if "dataset" not in pkl_data.keys() or pkl_data["dataset"] != dataset_name: 
                    pkl_data["dataset"] = dataset_name
                
                item_id = pkl_data['name']
                if item_id in lvis_item_dict.keys():
                    data_type = lvis_item_dict[item_id]
                    if "label" not in pkl_data.keys() or pkl_data["label"] != data_type:
                        pkl_data["label"] = data_type
                    if "label_count" not in pkl_data.keys() or pkl_data["label_count"] != class_to_num[data_type]:
                        pkl_data["label_count"] = int(class_to_num[data_type])
                else:
                    pkl_data["label"] = ""
                    pkl_data["label_count"] = -1

                torch.save(pkl_data, pkl_path)

    root = "/path/to/your/gaussian-splatting/clip3/objaverse_all/objects"
    with open( "//path/to/your/objarverse/label.json", "r") as file:
        class_to_num = json.load(file)
    
    lvis_item_dict = {}
    lvis_json_path = "//path/to/your/objarverse/lvis-annotations.json"
    with open(lvis_json_path, "r") as file:
        lvis_json = json.load(file)
    for split in lvis_json:
        for item_id in lvis_json[split]:
            lvis_item_dict[item_id]=split.lower()

    dataset_name="objaverse"
    add_pkl_attribute(root, dataset_name, class_to_num, lvis_item_dict)

if __name__ == "__main__":
    # transfer_abo()
    # transfer_objaverse()
    # transfer_mvimgnet()
    transfer_sunrgbd()