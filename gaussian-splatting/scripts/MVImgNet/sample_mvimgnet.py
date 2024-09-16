import os
from math import ceil

'''
CUDA_VISIBLE_DEVICES=0 python train.py -s data/mv_test/0a00be23 --eval --iterations 3000 --sh_degree 0 --output_recenter --use_mask --densify_from_iter 100 --model_path ./output/0a00be23
'''

root = "/path/to/your/gaussian-splatting/data/mvimgnet"
sh_root = os.path.join(root, "sh")
os.makedirs(sh_root, exist_ok=True)

output_root = "/path/to/your/gaussian-splatting/output/mvimgnet"
already_optimized_list = []
if os.path.exists(output_root):
    for data_type in os.listdir(output_root):
        for item_id in os.listdir(os.path.join(output_root, data_type)):
            if os.path.exists(os.path.join(output_root, data_type, item_id, "point_cloud", "iteration_500", "point_cloud.ply")):
                already_optimized_list.append(item_id)
print("already_optimized_list", len(already_optimized_list))

class_annotation_path = "//path/to/your/MVimgnet/scripts/mvimgnet_category.txt"
class_to_num = {}
with open(class_annotation_path, "r") as file:
    for line in file.readlines():
        line = line.replace("\n","").split(",")
        class_to_num[line[1].lower()] = line[0]

selected_items = []
data_type_dict = {}
for data_type in os.listdir(root):
    if data_type == "sh": continue
    data_type_root = os.path.join(root, data_type)

    for item_id in os.listdir(data_type_root):
        if item_id not in already_optimized_list:
            selected_items.append(os.path.join(data_type,item_id))
        data_type_dict[item_id] = data_type

samples_per_gpu = 10
gpus = [1,2,3,4,5]

# print(selected_items, len(selected_items))
# root = "//path/to/your/MVimgnet/mvi"
# for item in selected_items:
#     item = item.split("/")
#     data_type = class_to_num[item[0]]
#     if not os.path.exists(os.path.join(root, data_type, item[1], "sparse", "0", "images.bin")):
#         print("'"+os.path.join("/path/to/your/gaussian-splatting/data/mvimgnet", item[0], item[1])+"'")
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
        selected_gpu = gpus[item_count % len(gpus)]
        
        subfix = " > /dev/null 2>&1 &"
        if j == len(split_items) - 1: subfix=""

        content.append(
            "CUDA_VISIBLE_DEVICES="+str(selected_gpu)+" python train.py -s \'data/mvimgnet/"+item_id+"\'"+ \
            " --eval --update_step 50 --iterations 500 --sh_degree 0 --fix_nums 1024 --output_recenter --skip_test --use_mask --densify_from_iter 99 --model_path \'./output/mvimgnet/"+ item_id +"\'"+ subfix + "\n"
        )

        item_count += 1
    
    with open(os.path.join(sh_root, "run_"+str(i+1)+".sh"), "w") as file:
        file.writelines(content)
    sh_item_count += 1

print("sh_item_count: ", sh_item_count)

# 10 2min