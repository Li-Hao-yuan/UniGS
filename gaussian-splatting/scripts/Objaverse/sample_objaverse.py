import os
import numpy as np
import json
import random
from math import ceil
from tqdm import tqdm


'''
CUDA_VISIBLE_DEVICES=0 python train.py -s data/objaverse/64a0a475fefd46fbabd0611016329978 --eval --init_pt 1024 --fix_nums 1024 --iterations 5000 --densification_interval 501 --model_path ./output/obj
'''

samples_per_gpu = 12
gpus = [1, 2, 3, 4, 5]
save_folder_name = "objaverse"
skip_already_optimized = True
# skip_manchine_prompt = False
max_item_count = 92160

# 1day x 5gpu = 92160

root = "/path/to/your/gaussian-splatting/data/objaverse/"
sh_root = os.path.join(root, "sh")
os.makedirs(sh_root, exist_ok=True)

output_root = "/path/to/your/gaussian-splatting/output/objaverse"
already_optimized_list = []
for item_id in os.listdir(output_root):
    if os.path.isdir(os.path.join(output_root, item_id)):
        already_optimized_list.append(item_id)

transfer_root = "/path/to/your/gaussian-splatting/data/objaverse_transfer"
if os.path.exists(transfer_root):
    already_optimized_list.extend(os.listdir(transfer_root))

prompt_human_path = "//path/to/your/objarverse/Cap3D_human_Objaverse.pkl"
human_prompts = np.load(prompt_human_path, allow_pickle=True)

selected_items, ptr = [], 0
already_optimized_list.sort()

item_id_list = os.listdir(root)
# with open("//path/to/your/objarverse/lvis_items.json", "r") as file:
#     item_id_list = json.load(file)

if "sh" in item_id_list: item_id_list.remove("sh")
if "file_paths.json" in item_id_list: item_id_list.remove("file_paths.json")
item_id_list.sort()

for item_id in tqdm(already_optimized_list):
    while item_id_list[ptr]<item_id:
        selected_items.append(item_id_list[ptr])
        ptr += 1
        if ptr == len(item_id_list): break
    if item_id_list[ptr]==item_id:
        ptr += 1
    if ptr == len(item_id_list): break
if ptr < len(item_id_list): selected_items.extend(item_id_list[ptr:])

if len(selected_items) < max_item_count: 
    print("not enough items! %d - %d"%(len(selected_items),max_item_count))
    exit()
else: selected_items = selected_items[:max_item_count]

# print("selected_items", len(selected_items))
# exit()

sh_item_count = 0
items_each_sh = samples_per_gpu * len(gpus)
blocks = ceil(len(selected_items) / items_each_sh)

for i in range(blocks):
    split_items = selected_items[i*items_each_sh: (i+1)*items_each_sh]

    content = []
    item_count = 0
    for j, item in enumerate(split_items):
        item_id = item
        selected_gpu = gpus[item_count % len(gpus)] # + 1
        
        subfix = " > /dev/null 2>&1 &"
        if j == len(split_items) - 1: subfix=""

        content.append(
            "CUDA_VISIBLE_DEVICES="+str(selected_gpu)+" python train.py -s data/objaverse/"+item_id+ \
            " --eval --update_step 100 --iterations 1500 --sh_degree 0 --init_pt 1024 --fix_nums 1024 --densification_interval 501 --model_path ./output/"+save_folder_name+"/"+ item_id + subfix + "\n"
        )

        item_count += 1
    
    sh_item_count += 1
    with open(os.path.join(sh_root, "run_"+str(i+1)+".sh"), "w") as file:
        file.writelines(content)

print("sh_item_count: ", sh_item_count)