import os
import random
import json
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import sys

parser = ArgumentParser(description="Converting parameters")
parser.add_argument('--json_path', type=str, default=None)
args = parser.parse_args(sys.argv[1:])

if args.json_path is not None:
    json_path = args.json_path
    with open(json_path, "r") as file:
        content = json.load(file)["train"]

    item_path_list = []
    data_root = "//path/to/your/ABO/render_images_256"
    for item_path in content:
        item_path_list.append(
            os.path.join(data_root, item_path)
        )
else:
    item_path_list = []
    data_dir = "//path/to/your/ABO/render_images_256"
    for item_type in os.listdir(data_dir):
        item_type_root = os.path.join(data_dir, item_type)
        for item_id in os.listdir(item_type_root):
            item_path_list.append( os.path.join(item_type_root, item_id) )

    # specific
    # item_path_list = [
    #     "//path/to/your/ABO/render_images_256/chair/B07DBGHXWQ"
    # ]

data_ratio = [0.75,0.0,0.25]
camera_keys = ["up","down"]
# camera_keys = ["up"]
save_root = "/path/to/your/gaussian-splatting/data/ABO"
pts_root = "//path/to/your/ABO/3dmodels/xyz"
for item_path in tqdm(item_path_list):
    item_id = item_path.split("/")[-1]
    object_root = os.path.join(save_root, item_id)
    if os.path.exists(object_root): continue

    os.makedirs( os.path.join(object_root,"train"), exist_ok=True )
    os.makedirs( os.path.join(object_root,"val"), exist_ok=True )
    os.makedirs( os.path.join(object_root,"test"), exist_ok=True )

    pts_save_path = os.path.join(object_root, "pts.npy")
    pts_from_path = os.path.join(pts_root, item_path.split("/")[-2], item_id+".npy")
    if os.path.exists(pts_from_path) and not os.path.exists(pts_save_path):
        os.symlink(pts_from_path, pts_save_path)

    for camera_key in camera_keys:
        camera_data_root = os.path.join(item_path, camera_key)

        # shuffle
        file_list = os.listdir(camera_data_root)
        file_list.remove("transforms.json")
        random.shuffle( file_list )

        # sample
        train_sample_ptr = int(len(file_list)*data_ratio[0])
        val_sample_ptr = int(len(file_list)*data_ratio[1]) + train_sample_ptr

        train_path_list = file_list[:train_sample_ptr]
        val_path_list = file_list[train_sample_ptr:val_sample_ptr]
        test_path_list = file_list[val_sample_ptr:]

        train_path_list.sort()
        val_path_list.sort()
        test_path_list.sort()
        split_path_list = [train_path_list,val_path_list,test_path_list]

        # conver json
        camera_json_path = os.path.join(camera_data_root,"transforms.json")
        with open(camera_json_path,"r",encoding="utf-8") as file:
            json_info = json.load(file)

        split_path_name = ["train","val","test"]
        json_split_store = [{},{},{}]
        for i,single_path_list in enumerate(split_path_list):
            frames = []
            for image_path in single_path_list:
                image_path = image_path.split(".")
                count = image_path[1] if len(image_path) > 2 else "000"
                single_json_info = json_info[int(count)]

                new_json_info = {}
                # new_json_info["file_path"] = "./"+split_path_name[i]+"/"+camera_key+"_"+single_json_info["img_name"][:-4]
                new_json_info["file_path"] = "./"+split_path_name[i]+"/"+camera_key+"_"+count
                new_json_info["rotation"] = -1
                new_json_info["transform_matrix"] = single_json_info["M_world2cam"]

                frames.append( new_json_info )

            # save
            json_split_store[i]["camera_angle_x"] = 0.5212047834946819
            json_split_store[i]["frames"] = frames
            if os.path.exists(os.path.join(object_root,"transforms_"+split_path_name[i]+".json")):
                with open( os.path.join(object_root,"transforms_"+split_path_name[i]+".json"),"r") as file:    
                    tem_json_info = json.load(file)
                    json_split_store[i]["frames"].extend(tem_json_info["frames"])
            with open( os.path.join(object_root,"transforms_"+split_path_name[i]+".json"),"w") as file:
                json.dump(json_split_store[i], file, indent=4)
            
        for i,single_path_list in enumerate(split_path_list):
            for image_path in single_path_list:
                # os.system("cp "+os.path.join(camera_data_root,image_path)+" "+os.path.join(object_root,split_path_name[i],camera_key+"_"+image_path))

                image_path_list = image_path.split(".")
                count = image_path_list[1] if len(image_path_list) > 2 else "000"
                # print(os.path.join(camera_data_root,image_path), os.path.join(object_root,split_path_name[i],camera_key+"_"+count+".png"))
                os.symlink(os.path.join(camera_data_root,image_path), os.path.join(object_root,split_path_name[i],camera_key+"_"+count+".png"))

    # exit()
