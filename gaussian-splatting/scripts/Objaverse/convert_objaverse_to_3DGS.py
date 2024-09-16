import os
import json
import time
import random
import numpy as np
from tqdm import tqdm
import math
import open3d as o3d
from plyfile import PlyData, PlyElement
import datetime
import clip

# python /path/to/your/gaussian-splatting/scripts/Objaverse/convert_objaverse_to_3DGS.py
# nohup python /path/to/your/gaussian-splatting/scripts/Objaverse/convert_objaverse_to_3DGS.py > "/path/to/your/gaussian-splatting/scripts/Objaverse/convert.log" 2>&1 &

# 30min -> 5s

# imgs: 798759
# pts: 5000*160=800_000 | 798470
# now : 765987


# read csv
def get_item_prompt(prompt_csv_path):
    item_prompt_dict = {}
    if prompt_csv_path.endswith("csv"):
        with open(prompt_csv_path, "r") as file:
            for line in file.readlines():
                split_index = line.index(",")

                item_id = line[:split_index]
                prompt = line[split_index+1:].replace("\"","")

                item_prompt_dict[item_id] = prompt
    else:
        item_prompt_dict = np.load(prompt_csv_path, allow_pickle=True)
    
    return item_prompt_dict

pts_root = "//path/to/your/objarverse/glbs"
print("Load from glb model!", flush=True)

# pts_root = "//path/to/your/objarverse/openshape/objaverse-processed/merged_for_training_final/Objaverse"
# print("Load from point cloud!", flush=True)

img_root = "//path/to/your/objarverse/views_release"
# prompt_machine_path = "//path/to/your/objarverse/Cap3D_automated_Objaverse_full.csv"
prompt_machine_path = "//path/to/your/objarverse/Cap3D_automated_Objaverse_full_no3Dword.csv"
prompt_human_path = "//path/to/your/objarverse/Cap3D_human_Objaverse.pkl"
save_root = "/path/to/your/gaussian-splatting/data/objaverse"
# save_root = "/path/to/your/gaussian-splatting/data/"
os.makedirs(save_root, exist_ok=True)

prompt_machine_dict = get_item_prompt(prompt_machine_path)
prompt_human_dict = get_item_prompt(prompt_human_path)

# already model
choose_item_list, choose_item_dict = [], {}
save_json_path = os.path.join(save_root, "file_paths.json")
if os.path.exists(save_json_path):
    with open(save_json_path, "r") as file:
        choose_item_dict = json.load(file)
        choose_item_list = list(choose_item_dict.keys())
choose_item_list.sort()

update_choosed_item_list = []
# for item_id in tqdm(os.listdir(save_root),desc="scanning completed model: "):
#     save_item_root = os.path.join(save_root, item_id)
#     train_json_path, test_json_path = os.path.join(save_item_root, "transforms_train.json"), os.path.join(save_item_root, "transforms_test.json")
#     if (item_id not in choose_item_list) and os.path.exists(train_json_path) and os.path.exists(test_json_path):
#         update_choosed_item_list.append(item_id)  
# print("update_choosed_item_list : %d"%(len(update_choosed_item_list)))

# exist pts
item_id_list, pts_path_dict = [], {}
for split in tqdm(os.listdir(pts_root), desc="scanning 3d model: "):

    split_root = os.path.join(pts_root, split)
    split_item_id_list = os.listdir(split_root)
    split_item_id_list.sort()

    ptr = 0
    for item_id in choose_item_list:
        split_item_id = split_item_id_list[ptr].split(".")[0]
        while split_item_id<item_id:
            item_id_list.append(split_item_id)
            pts_path_dict[split_item_id] = os.path.join(split_root, split_item_id_list[ptr])
            ptr += 1
            if ptr>=len(split_item_id_list): break
            split_item_id = split_item_id_list[ptr].split(".")[0]
        if split_item_id == item_id: ptr += 1
        if ptr>=len(split_item_id_list): break

# print("Rechecking...")
# item_id_list.sort()
# choose_item_list.sort()
# item_id_list_ptr = 0
# for item_id in tqdm(choose_item_list):
#     if item_id_list[item_id_list_ptr] < item_id:
#         item_id_list_ptr += 1
#         if item_id_list_ptr >= len(item_id_list): break
#     elif item_id_list[item_id_list_ptr] == item_id:
#         raise

# exist model img
render_item_list, ptr = [], 0
exist_model_list = os.listdir(img_root)
exist_model_list.sort()
for item_id in tqdm(choose_item_list, desc="scanning render imgs: "):
    while exist_model_list[ptr]<item_id:
        render_item_list.append(exist_model_list[ptr])
        ptr += 1
        if ptr>=len(exist_model_list): break
    if exist_model_list[ptr] == item_id: ptr += 1
    if ptr>=len(exist_model_list): break

# print("Rechecking...")
# render_item_list.sort()
# choose_item_list.sort()
# item_id_list_ptr = 0
# for item_id in tqdm(choose_item_list):
#     if render_item_list[item_id_list_ptr] < item_id:
#         item_id_list_ptr += 1
#         if item_id_list_ptr >= len(render_item_list): break
#     elif render_item_list[item_id_list_ptr] == item_id:
#         raise

real_item_id_list, real_item_prompt_dict = [], {}
render_ptr, item_ptr, machine_ptr, human_ptr  = 0, 0, 0, 0
prompt_machine_list = list(prompt_machine_dict.keys())
prompt_human_list = list(prompt_human_dict.keys())

render_item_list.sort()
item_id_list.sort()
prompt_machine_list.sort()
prompt_human_list.sort()

print("\nMatching...")
while True:
    if render_ptr >= len(render_item_list): break
    if item_ptr >= len(item_id_list): break

    render_item = render_item_list[render_ptr]
    glb_item = item_id_list[item_ptr]

    if render_item > glb_item: item_ptr +=1
    elif render_item < glb_item: render_ptr += 1

    if render_item == glb_item:
        real_item_prompt_dict[render_item] = {"machine":False, "human":False}
        while machine_ptr < len(prompt_machine_list):
            machine_prompt_item_id = prompt_machine_list[machine_ptr]
            if machine_prompt_item_id < render_item: machine_ptr += 1
            elif machine_prompt_item_id == render_item:
                real_item_prompt_dict[render_item]["machine"] = True
                machine_ptr += 1
            elif machine_prompt_item_id > render_item:
                break

        while human_ptr < len(prompt_human_list):
            human_prompt_item_id = prompt_human_list[human_ptr]
            if human_prompt_item_id < render_item: human_ptr += 1
            elif human_prompt_item_id == render_item:
                real_item_prompt_dict[render_item]["human"] = True
                human_ptr += 1
            elif human_prompt_item_id > render_item:
                break

        real_item_id_list.append(render_item)

        render_ptr += 1
        item_ptr += 1
print("real_item_id_list: ", len(real_item_id_list), flush=True)

train_imgs = 12
default_focal = 560
default_resolution = 512
fov = 2*math.atan(default_resolution/(2*default_focal))
transform_matrix = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]])

selected_item_count, count = 900000, 0
# selected_item_count, count = 1, 0 

time_begin = time.perf_counter()
selected_item_count -= len(choose_item_list)
random.shuffle(real_item_id_list)
for i, item_id in enumerate(real_item_id_list):
    # if item_id=="zicHab4DIw41ZXdXHMYVogs7RXm": continue

    now = time.perf_counter()
    date = datetime.timedelta(seconds=now-time_begin)
    time_left = datetime.timedelta(seconds=(now-time_begin)/(i+1)*len(real_item_id_list) - (now-time_begin))
    if (i+1)%10 == 0:
        print("Time %s | %s, Scanning count : %d|%d - %d|%d"%(date, time_left, count, selected_item_count, i, len(real_item_id_list)), flush=True)

    # prompt of Cap3D
    # machine_prompt_exist_flag = real_item_prompt_dict[item_id]["machine"]
    # human_prompt_exist_flag = real_item_prompt_dict[item_id]["human"]

    machine_prompt_exist_flag = item_id in prompt_machine_dict.keys()
    human_prompt_exist_flag = item_id in prompt_human_dict.keys()

    # if not machine_prompt_exist_flag and not human_prompt_exist_flag: continue
    if len(os.listdir(os.path.join(img_root, item_id)))<24: continue

    save_item_root = os.path.join(save_root, item_id)

    if machine_prompt_exist_flag:
        machine_prompt = prompt_machine_dict[item_id]
        if not isinstance(machine_prompt, str): machine_prompt = machine_prompt[0]
        machine_prompt = machine_prompt.replace("\n","").lower()

        try: clip.tokenize(machine_prompt)
        except: continue
    else: machine_prompt = ""

    # human prompt
    if human_prompt_exist_flag:
        human_prompt = prompt_human_dict[item_id]
        if not isinstance(human_prompt, list): human_prompt = [human_prompt]
        if human_prompt == ["not clear"]: 
            human_prompt = [""]
        else:

            for i in range(len(human_prompt)):
                if human_prompt == "not clear": continue
                human_prompt[i] = human_prompt[i].replace("\n","").lower()

                try: clip.tokenize(human_prompt[i])
                except: continue
    else: human_prompt = [""]

    # file paths
    choose_item_dict[item_id] = {
        "pts_path": pts_path_dict[item_id],
        "img_path": os.path.join(img_root, item_id),
        "prompt_machine": machine_prompt,
        "prompt_human": human_prompt,
    }
    if item_id in update_choosed_item_list: continue

    # pts path
    item_pts_path = choose_item_dict[item_id]["pts_path"]
    save_npy_path = os.path.join(save_item_root, "pts.npy")
    os.makedirs(save_item_root, exist_ok=True)

    if item_pts_path.endswith(".glb"):

        # ply
        try:
            save_ply_path = os.path.join(save_item_root, "points3d.ply")
            glb_item = o3d.io.read_triangle_mesh(item_pts_path)
            glb_pts = glb_item.sample_points_uniformly(number_of_points=100_000)
        except:
            continue
        
        o3d.io.write_point_cloud( save_ply_path , glb_pts)

        # xyz
        plydata = PlyData.read(save_ply_path)
        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            -np.asarray(plydata.elements[0]["z"]),
            np.asarray(plydata.elements[0]["y"])),  axis=1)
    else:

        pts_data = np.load(item_pts_path, allow_pickle=True)[()]

        if not machine_prompt_exist_flag and not human_prompt_exist_flag:
            machine_prompt = pts_data["blip_caption"]

        xyz = pts_data["xyz"]
        x = xyz[:,0:1].copy()
        y = xyz[:,1:2].copy()
        z = xyz[:,2:3].copy()
        xyz = np.concatenate((-x,z,y), axis=-1)

    min_xyz = np.array([np.min(xyz[:,0]), np.min(xyz[:,1]), np.min(xyz[:,2])])
    max_xyz = np.array([np.max(xyz[:,0]), np.max(xyz[:,1]), np.max(xyz[:,2])])
    scale = 1/max(max_xyz-min_xyz)
    xyz *= scale
    min_xyz = np.array([np.min(xyz[:,0]), np.min(xyz[:,1]), np.min(xyz[:,2])])
    max_xyz = np.array([np.max(xyz[:,0]), np.max(xyz[:,1]), np.max(xyz[:,2])])
    offset = -(min_xyz+max_xyz)/2
    xyz += offset

    np.save(save_npy_path, xyz)

    train_img_root, test_img_root = os.path.join(save_item_root, "train"), os.path.join(save_item_root, "test")
    train_json_path, test_json_path = os.path.join(save_item_root, "transforms_train.json"), os.path.join(save_item_root, "transforms_test.json")
    os.makedirs(train_img_root, exist_ok=True)
    os.makedirs(test_img_root, exist_ok=True)

    # imgs
    this_img_inds = [i for i in range(12)]
    random.shuffle(this_img_inds)
    train_img_inds, test_img_inds = this_img_inds[:train_imgs], this_img_inds[train_imgs:]
    train_json = {
        "camera_angle_x": fov,
        "frames":[]
    }
    test_json = {
        "camera_angle_x": fov,
        "frames":[]
    }
    for img_ind in range(12):
        count_name = count_name = (3-len(str(img_ind)))*"0"+str(img_ind)
        img_path = os.path.join(img_root, item_id, count_name+".png")

        pose = np.load(os.path.join(img_root, item_id, count_name+".npy"))
        pose = np.concatenate((pose, [[0,0,0,1]]),axis=0)
        pose = np.linalg.inv(pose)
        pose = np.round(pose, 5)

        if img_ind in train_img_inds:
            train_img_path = os.path.join(train_img_root, count_name+".png")
            if not os.path.exists(train_img_path): os.symlink(img_path, train_img_path)
            train_json["frames"].append({
                "file_path":"train/"+count_name,
                "transform_matrix": pose.tolist()
            })
        else:
            test_img_path = os.path.join(test_img_root, count_name+".png")
            if not os.path.exists(test_img_path):  os.symlink(img_path, test_img_path)
            test_json["frames"].append({
                "file_path":"test/"+count_name,
                "transform_matrix": pose.tolist()
            })
    with open(train_json_path, "w") as file:
        json.dump(train_json, file, indent=4)
    with open(test_json_path, "w") as file:
        json.dump(test_json, file, indent=4)

    choose_item_list.append(item_id)        
    count += 1
    
    if (i+1)%10000 == 0:
        print("Save json!", flush=True)
        with open(save_json_path, "w") as file:
            json.dump(choose_item_dict, file, indent=4)

    if count >= selected_item_count: break

print("selected item : %d|%d"%(count, selected_item_count), flush=True)

if count>0 or True:
    with open(save_json_path, "w") as file:
        json.dump(choose_item_dict, file, indent=4)


# 38 39 5 20 21 

