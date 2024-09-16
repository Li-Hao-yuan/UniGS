import os
import json
from pathlib import Path
from tqdm import tqdm

def transfer_dataset_by_type(data_dir):
    root = "//path/to/your/MVimgnet"
    class_annotation_path = "//path/to/your/MVimgnet/scripts/mvimgnet_category.txt"
    num_to_class = {}
    with open(class_annotation_path, "r") as file:
        for line in file.readlines():
            line = line.replace("\n","").split(",")
            num_to_class[line[0]] = line[1]

            data_type_root = os.path.join(data_dir, line[1])
            if not os.path.exists(data_type_root): os.makedirs(data_type_root)

    item_to_class = {}
    for folder_name in os.listdir(root):
        if folder_name[:4]=="mvi_":
            for data_type in os.listdir(os.path.join(root, folder_name)):
                scanned_data_type = num_to_class[data_type]

                for item_id in os.listdir(os.path.join(root, folder_name, data_type)):
                    item_to_class[item_id] = scanned_data_type
    
    for item_id in tqdm(os.listdir(data_dir)):
        if item_id in item_to_class.keys():
            item_type = item_to_class[item_id]
            # line = "cp -r" + " " + os.path.join(data_dir, item_id) + " " + "\'" + os.path.join(data_dir, item_type, item_id)+"\'"
            line = "mv" + " " + os.path.join(data_dir, item_id) + " " + "\'" + os.path.join(data_dir, item_type, item_id)+"\'"
            os.system(line)

def merge_datasets(dataset_paths, save_root):
    os.makedirs(save_root, exist_ok=False)

    file_paths, sample_all = None, None
    for dataset_path in dataset_paths:
        
        with open(os.path.join(dataset_path, "file_paths.json"), "r") as file:
            file_paths_content = json.load(file)
        with open(os.path.join(dataset_path, "sample_all.json"), "r") as file:
            sample_all_content = json.load(file)
        if file_paths is None: 
            file_paths = file_paths_content
            sample_all = sample_all_content
        else:
            file_paths["original"]["train"].extend(file_paths_content["original"]["train"])
            file_paths["original"]["test"].extend(file_paths_content["original"]["test"])
            file_paths["original"]["val"].extend(file_paths_content["original"]["val"])

            sample_all["train"].extend(sample_all_content["train"])
            sample_all["test"].extend(sample_all_content["test"])

            for key in file_paths_content["all"].keys():
                if key not in file_paths["all"].keys():
                    file_paths["all"][key] = file_paths_content["all"][key]
                else:
                    file_paths["all"][key].extend(file_paths_content["all"][key])

        dataset_objects_root = os.path.join(dataset_path, "objects")
        for data_type in tqdm(os.listdir(dataset_objects_root)):
            data_type_root = os.path.join(dataset_path, data_type)
            save_data_type_root = os.path.join(save_root, data_type)
            if not os.path.isdir(data_type_root): continue

            if not os.path.exists(save_data_type_root):
                os.makedirs(save_data_type_root)
                exist_count = 0
            else:
                exist_count = len(list(Path(save_data_type_root).glob("*")))

            for pkl_name in os.listdir(data_type_root):
                os.symlink(os.path.join(data_type_root, pkl_name), os.path.join(save_data_type_root, str(exist_count)+".pkl"))
                exist_count += 1
            
    with open(os.path.join(save_root, "file_paths.json"), "w") as file:
        json.dump(file_paths, file, indent=4)
    with open(os.path.join(save_root, "sample_all.json"), "w") as file:
        json.dump(sample_all, file, indent=4)
    

# transfer_dataset_by_type("/path/to/your/gaussian-splatting/output/mvimgnet")

merge_datasets(
    ["/path/to/your/gaussian-splatting/clip3/mvimgnet_500",
     "/path/to/your/gaussian-splatting/clip3/sunrgbd_all"],
   save_root="/path/to/your/gaussian-splatting/clip3/sun_mvi"  
)