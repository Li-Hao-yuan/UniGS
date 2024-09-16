import os
from math import ceil

'''
CUDA_VISIBLE_DEVICES=0 python train.py -s data/SUNRGBD/realsense/lg/2014_10_24-09_14_34-1311000073 --eval --fix_pts --opacity_reset_interval 50 --iterations 500 --model_path ./output/SUN/bfx_reset
'''

root = "/path/to/your/gaussian-splatting/data/SUNRGBD"
sh_root = os.path.join(root, "sh")
os.makedirs(sh_root, exist_ok=True) 

selected_items = []
for folder_name1 in os.listdir(root):
    if folder_name1 == "sh": continue
    folder_root_1 = os.path.join(root, folder_name1)

    for folder_name2 in os.listdir(folder_root_1):
        folder_root_2 = os.path.join(folder_root_1, folder_name2)

        if folder_name2 == "sun3ddata":
            for folder_name3 in os.listdir(folder_root_2):
                folder_root_3 = os.path.join(folder_root_2, folder_name3)

                for folder_name4 in os.listdir(folder_root_3):
                    folder_root_4 = os.path.join(folder_root_3, folder_name4)

                    for item_name in os.listdir(folder_root_4):
                        selected_items.append(os.path.join(folder_name1, folder_name2, folder_name3, folder_name4, item_name))

        else:
            for item_name in os.listdir(folder_root_2):
                selected_items.append(os.path.join(folder_name1, folder_name2, item_name))

samples_per_gpu = 10
gpus = 6

sh_item_count = 0
items_each_sh = samples_per_gpu * gpus
blocks = ceil(len(selected_items) / items_each_sh)

for i in range(blocks):
    split_items = selected_items[i*items_each_sh: (i+1)*items_each_sh]

    content = []
    item_count = 0
    for j, item in enumerate(split_items):
        # item_id = item.split("/")[-1]
        item_id = item
        selected_gpu = item_count % gpus
        
        subfix = " > /dev/null 2>&1 &"
        if j == len(split_items) - 1: subfix=""

        sh_item_count += 1
        content.append(
            "CUDA_VISIBLE_DEVICES="+str(selected_gpu)+" python train.py -s data/SUNRGBD/"+item_id+ \
            " --eval --fix_pts --skip_test --sh_degree 0 --opacity_reset_interval 50 --iterations 500 --model_path ./output/SUNRGBD/"+ item_id + subfix + "\n"
        )

        item_count += 1
    
    with open(os.path.join(sh_root, "run_"+str(i+1)+".sh"), "w") as file:
        file.writelines(content)

print("sh_item_count: ", sh_item_count)
# run 
#
#
# 