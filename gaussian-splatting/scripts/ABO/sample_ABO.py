import os
import numpy as np
import json
import random
from math import ceil

if_sample = False
if_convert = False
if_gen_sh = True
if_run_sh = False

## sample
json_path = "/path/to/your/gaussian-splatting/data/ABO/ABO_3DGS.json"
data_root = "//path/to/your/ABO/render_images_256"

# if if_sample:
#     selected_items, selected_item_nums = [], 0
#     for data_type in os.listdir(data_root):
#         if data_type.lower() in ["chair","freestanding_shelter"]: continue

#         item_names = os.listdir( os.path.join(data_root, data_type) )
#         if len(item_names) < 10: continue
#         random.shuffle(item_names)
#         item_names = item_names[:int(len(item_names)/10)]

#         selected_item_nums += len(item_names)
#         for item_name in item_names:
#             selected_items.append(data_type+"/"+item_name)

#     print("selected_item_nums exclude chairs : ",selected_item_nums)

#     item_names = os.listdir(os.path.join(data_root, "chair"))
#     random.shuffle(item_names)
#     item_names = item_names[:1000-selected_item_nums]
#     for item_name in item_names:
#             selected_items.append("chair/"+item_name)

#     with open(json_path, "w") as file:
#         json.dump({"train":selected_items}, file, indent=4)
# else:
#     with open(json_path, "r") as file:
#         selected_items = json.load(file)["train"]

data_root = "/path/to/your/gaussian-splatting/data/ABO"
selected_items = os.listdir(data_root)
if "sh" in selected_items: selected_items.remove("sh")
if "file_paths.json" in selected_items: selected_items.remove("file_paths.json")

## convert
convert_file_path = "/path/to/your/gaussian-splatting/scripts/convert_ABO_to_DVGO.py"
if if_convert:
    os.system(
        "python" + " " + convert_file_path + " " + "--json_path" + " " + json_path
    )


## sh
sh_root = "/path/to/your/gaussian-splatting/data/ABO/sh/"
if if_gen_sh:
    '''
    CUDA_VISIBLE_DEVICES=0 python train.py -s data/ABO/B07VPNX3QL --eval --init_pt 1024 --fix_nums 1024 --iterations 50000 --densification_interval 501 --model_path ./output/ABO/B07VPNX3QL > /dev/null 2>&1 &
    '''
    os.makedirs(sh_root, exist_ok=True) 

    samples_per_gpu = 15
    gpus = [1, 2, 3, 4, 5]

    items_each_sh = samples_per_gpu * len(gpus)
    blocks = ceil(len(selected_items) / items_each_sh)

    for i in range(blocks):
        split_items = selected_items[i*items_each_sh: (i+1)*items_each_sh]

        content = []
        item_count = 0
        for j, item in enumerate(split_items):
            item_id = item.split("/")[-1]
            selected_gpu = gpus[item_count % len(gpus)]
            
            subfix = " > /dev/null 2>&1 &"
            if j == len(split_items) - 1: subfix=""

            content.append(
                "CUDA_VISIBLE_DEVICES="+str(selected_gpu)+" python train.py -s data/ABO/"+item_id+ \
                " --eval --update_step 100 --skip_test --sh_degree 0 --init_pt 1024 --fix_nums 1024 --iterations 2000 --densification_interval 501 --model_path ./output/ABO/"+ item_id + subfix + "\n"
            )

            item_count += 1
        
        with open(os.path.join(sh_root, "run_"+str(i+1)+".sh"), "w") as file:
            file.writelines(content)