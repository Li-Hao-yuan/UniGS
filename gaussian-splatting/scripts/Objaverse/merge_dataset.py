import os
import json
from pathlib import Path
from tqdm import tqdm

def merge_datasets(dataset_paths, save_root, keep_first_json=True):
    os.makedirs(save_root, exist_ok=False)
    os.makedirs(os.path.join(save_root, "objects"), exist_ok=False)

    file_paths, sample_all = None, None
    for dataset_path in dataset_paths:
        
        with open(os.path.join(dataset_path, "file_paths.json"), "r") as file:
            file_paths_content = json.load(file)
        with open(os.path.join(dataset_path, "sample_all.json"), "r") as file:
            sample_all_content = json.load(file)

        # file paths
        if file_paths is None: file_paths = file_paths_content
        else:
            if keep_first_json: file_paths["original"]["train"].extend(file_paths_content["original"]["test"])
            else: file_paths["original"]["test"].extend(file_paths_content["original"]["test"])

            file_paths["original"]["train"].extend(file_paths_content["original"]["train"])
            file_paths["original"]["val"].extend(file_paths_content["original"]["val"])

            if isinstance(file_paths_content["all"], dict):
                for key in file_paths_content["all"].keys():
                    file_paths["all"].extend(file_paths_content["all"][key])

        # sample all
        if sample_all is None: sample_all = sample_all_content
        else:
            if keep_first_json: sample_all["train"].extend(sample_all_content["test"])
            else: sample_all["test"].extend(sample_all_content["test"])

            sample_all["train"].extend(sample_all_content["train"])
            # sample_all["all"].extend(sample_all_content["all"])

        # obj
        dataset_objects_root = os.path.join(dataset_path, "objects")
        for data_type in tqdm(os.listdir(dataset_objects_root)):
            if data_type.endswith(".json"): continue

            data_type_root = os.path.join(dataset_path, "objects", data_type)
            if not os.path.isdir(data_type_root): 
                os.symlink(data_type_root, os.path.join(save_root, "objects", data_type))
            else:

                for pkl_name in os.listdir(data_type_root):
                    os.symlink(os.path.join(data_type_root, pkl_name), os.path.join(save_root, "objects", data_type+"_"+pkl_name))
            
    with open(os.path.join(save_root, "file_paths.json"), "w") as file:
        json.dump(file_paths, file, indent=4)
    with open(os.path.join(save_root, "sample_all.json"), "w") as file:
        json.dump(sample_all, file, indent=4)
    


merge_datasets(
    ["/path/to/your/gaussian-splatting/clip3/objaverse_all",
     "/path/to/your/gaussian-splatting/clip3/sunrgbd_all",],
   save_root="/path/to/your/gaussian-splatting/clip3/objaverse_sun"  
)