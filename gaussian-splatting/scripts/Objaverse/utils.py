import os
import json
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image

def check_empty_3dgs_data():
    save_root = "/path/to/your/gaussian-splatting/data/objaverse"
    selected_keys = []
    for item_id in os.listdir(save_root):
        if item_id == "file_paths.json": continue
        selected_keys.append(item_id)
        train_img_root = os.path.join(save_root, item_id, "train")

        if len(os.listdir(train_img_root))<12:
            os.system("rm -r "+os.path.join(save_root, item_id))

def check_empty_3dgs_results(file_path_json):
    def check_empty(path_list):
        if isinstance(path_list, dict):
            for key in path_list.keys():
                path_list[key] = check_empty(path_list[key])
        elif isinstance(path_list, list):
            exist_list = []
            for path in path_list:
                if os.path.exists(path): exist_list.append(path)
            path_list = exist_list
        return path_list

    with open(file_path_json, "r") as file:
        file_paths = json.load(file)
    new_file_paths = check_empty(file_paths)

    with open(file_path_json, "w") as file:
        json.dump(new_file_paths, file, indent=4)
    
def find_objaverse_of_human_prompt():
    glb_root = "//path/to/your/objarverse/glbs"
    img_root = "//path/to/your/objarverse/views_release"
    prompt_human_path = "//path/to/your/objarverse/Cap3D_human_Objaverse.pkl"
    # save_root = "/path/to/your/gaussian-splatting/data/objaverse"

    # exist glb model
    glb_model_list = []
    for split in tqdm(os.listdir(glb_root), desc="scanning 3d model: "):
        split_root = os.path.join(glb_root, split)
        for item_name in os.listdir(split_root):
            item_id = item_name.split(".")[0]
            glb_model_list.append(item_id)

    # exist model img
    render_item_list = []
    for item_id in tqdm(os.listdir(img_root), desc="scanning render imgs: "):
        render_item_list.append(item_id)

    scanning_glb_items = []
    human_prompts = np.load(prompt_human_path, allow_pickle=True)
    for item_id in tqdm(human_prompts.keys(), desc="scanning item ids"):
        if item_id in render_item_list and item_id not in glb_model_list:
            scanning_glb_items.append(item_id)

    with open("/path/to/your/gaussian-splatting/scripts/Objaverse/human_download.json", "w") as file:
        json.dump({"train":scanning_glb_items}, file, indent=4)

def find_objaverse():
    glb_root = "//path/to/your/objarverse/glbs"
    img_root = "//path/to/your/objarverse/views_release"
    pts_root = "//path/to/your/objarverse/openshape/objaverse-processed/merged_for_training_final/Objaverse"
    # prompt_human_path = "//path/to/your/objarverse/Cap3D_human_Objaverse.pkl"
    # save_root = "/path/to/your/gaussian-splatting/data/objaverse"

    # exist model img
    render_item_list = []
    for item_id in tqdm(os.listdir(img_root), desc="scanning render imgs: "):
        render_item_list.append(item_id)
    
    # exist glb model
    glb_model_list = []
    for split in tqdm(os.listdir(glb_root), desc="scanning 3d model: "):
        split_root = os.path.join(glb_root, split)
        for item_name in os.listdir(split_root):
            item_id = item_name.split(".")[0]
            glb_model_list.append(item_id)

    # exist pts model
    pts_model_list = []
    for split in tqdm(os.listdir(pts_root), desc="scanning pts model: "):
        split_root = os.path.join(pts_root, split)
        for item_name in os.listdir(split_root):
            item_id = item_name.split(".")[0]
            pts_model_list.append(item_id)

    glb_model_list.extend(pts_model_list)
    glb_model_list = list(set(glb_model_list))

    render_item_list.sort()
    glb_model_list.sort()

    render_item_ptr = 0
    scanning_item_list = []

    for item_id in tqdm(glb_model_list, desc="scanning items: "):
        while render_item_list[render_item_ptr]<item_id:
            scanning_item_list.append(render_item_list[render_item_ptr])
            render_item_ptr += 1
            if render_item_ptr>=len(render_item_list): break
        if render_item_list[render_item_ptr] == item_id: render_item_ptr += 1
        if render_item_ptr>=len(render_item_list): break
    if render_item_ptr < len(render_item_list):
        scanning_item_list.extend(render_item_list[render_item_ptr:])
    print("Have imgs without model: %d"%(len(scanning_item_list)))

    scanning_item_list_inv = []
    glb_model_ptr = 0
    for item_id in tqdm(render_item_list, desc="scanning items inv: "):
        while glb_model_list[glb_model_ptr]<item_id:
            scanning_item_list_inv.append(glb_model_list[glb_model_ptr])
            glb_model_ptr += 1
            if glb_model_ptr>=len(glb_model_list): break
        if glb_model_list[glb_model_ptr] == item_id: glb_model_ptr += 1
        if glb_model_ptr>=len(glb_model_list): break
    if glb_model_ptr < len(glb_model_list):
        scanning_item_list_inv.extend(glb_model_list[glb_model_ptr:])
    print("Have model without imgs: %d"%(len(scanning_item_list_inv)))

    with open("/path/to/your/gaussian-splatting/scripts/Objaverse/to_download.json", "w") as file:
        json.dump(scanning_item_list, file, indent=4)

def count_objaverse_of_prompts():
    gs_data_root = "/path/to/your/gaussian-splatting/data/objaverse"
    prompt_human_path = "//path/to/your/objarverse/Cap3D_human_Objaverse.pkl"
    
    gs_data_items = os.listdir(gs_data_root)

    # 21604 / 39536
    scanning_glb_items = []
    human_prompts = np.load(prompt_human_path, allow_pickle=True)
    for item_id in tqdm(human_prompts.keys(), desc="scanning item ids"):
        if item_id in gs_data_items:
            scanning_glb_items.append(item_id)
    
    print(len(scanning_glb_items))

def dump_dataset_split():
    root = "/path/to/your/gaussian-splatting/output/objaverse"
    save_root = "/path/to/your/gaussian-splatting/scripts/Objaverse/split"

    item_id_list = []
    for item_id in os.listdir(root):
        if item_id == "sh": continue

        item_path = os.path.join(root, item_id)
        if not os.path.isdir(item_path): continue

        item_id_list.append(item_id)

    with open(os.path.join(save_root, "all.json"), "w") as file:
        json.dump(item_id_list, file, indent=4)

def recheck_dataset_split():
    root1 = "/path/to/your/gaussian-splatting/output/objaverse"
    root2 = "/path/to/your/gaussian-splatting/output/objaverse_human"

    item_id_list1 = os.listdir(root1)
    item_id_list2 = os.listdir(root2)

    count = 0
    for item_id in tqdm(item_id_list2):
        if item_id in item_id_list1:
            # print(item_id)
            count += 1
        else:
            os.system("mv"+" "+os.path.join(root2, item_id)+" "+os.path.join(root1, item_id))
    print(count)

def recheck_dataset():
    import clip


    root = "/path/to/your/gaussian-splatting/clip3/objaverse_all/objects"
    save_json_path = "/path/to/your/gaussian-splatting/scripts/Objaverse/split/recheck.json"
    if os.path.exists(save_json_path):
        with open(save_json_path, "r") as file:
            recheck_item_list = json.load(file)
    else:
        recheck_item_list = []
        with open(save_json_path, "w") as file:
            json.dump(recheck_item_list, file, indent=4)

    problems = [
        "cc0c7762bdd64b899a32262191945063",
        "57f0fa3da7cf4434b8bd30cd74b23dcf",
        "1e3329ebadac436ea12018d5e5e96c42",
        "5189790c9e5d44a58c9e25664b6e5601",
        "39aa4363bd7945658be5d1f28a3969f8",
        "ffddc991690446d399adb67792063d60",
        "1ff3edc7f424491785ddc12a47a42076",
        "3670044d4bfa4598836e7d0743248dcc",
        "c130a2b8c219469cb1e82ee190be67ee",
        "6d3ca7d8622444099d69cd5a4d273187",
        "e8cd1cc6825548f8a5a50f7ec22ae377",
        "22dd8551cb3c4016af99ed49ae2f041b",
    ]

    for pkl_name in tqdm(os.listdir(root)):
        if pkl_name.split(".")[0] in problems: continue
        if pkl_name in recheck_item_list: continue
        
        pkl_path = os.path.join(root, pkl_name)
        pkl_data = torch.load(pkl_path)

        if "human_prompt" in pkl_data.keys():
            try:
                clip.tokenize(pkl_data["human_prompt"][0])
                clip.tokenize(pkl_data["machine_prompt"][0])
                recheck_item_list.append(pkl_name)
            except:
                print(pkl_name)

        elif "machine_prompt" in pkl_data.keys():
            try:
                clip.tokenize(pkl_data["machine_prompt"][0])
                recheck_item_list.append(pkl_name)
            except:
                print(pkl_name)
        
        with open(save_json_path, "w") as file:
            json.dump(recheck_item_list, file, indent=4)

def compare_objaverse_of_not_complete_model():
    save_root = "/path/to/your/gaussian-splatting/data/objaverse"
    pts_root = "//path/to/your/objarverse/openshape/objaverse-processed/merged_for_training_final/Objaverse"

    complete_item_list = os.listdir(save_root)
    if "sh" in complete_item_list: complete_item_list.remove("sh")
    if "file_paths.json" in complete_item_list: complete_item_list.remove("file_paths.json")

    complete_item_list.sort()

    item_id_list, ptr = [], 0
    for split in tqdm(os.listdir(pts_root), desc="scanning pts model: "):

        split_root = os.path.join(pts_root, split)
        split_item_id_list = os.listdir(split_root)
        split_item_id_list.sort()

        ptr = 0
        for item_id in complete_item_list:
            split_item_id = split_item_id_list[ptr].split(".")[0]
            while split_item_id<item_id:
                item_id_list.append(split_item_id)
                ptr += 1
                if ptr>=len(split_item_id_list): break
                split_item_id = split_item_id_list[ptr].split(".")[0]
            if split_item_id == item_id: ptr += 1
            if ptr>=len(split_item_id_list): break

    print("item_id_list: ", len(item_id_list))

def double_check_dataset():
    output_root = "/path/to/your/gaussian-splatting/output/objaverse"

    lvis_json_path = "//path/to/your/objarverse/lvis_items.json"
    with open(lvis_json_path, "r") as file:
        lvis_items = json.load(file)

    item_id_list = []
    # for item_id in os.listdir(output_root):
    for item_id in lvis_items:
        exist_flag = False
        for iteration in [1500, 2000, 3000]:
            exist_flag = exist_flag or os.path.exists(os.path.join(output_root, item_id, "point_cloud", "iteration_"+str(iteration), "point_cloud.ply"))

        if not exist_flag: 
            print(item_id)
            item_id_list.append(os.path.join(output_root, item_id))
    
    print("item_id_list", len(item_id_list))
    # for item_path in tqdm(item_id_list, desc="Deleting :"):
    #     os.system("rm -r "+item_path)

def transfer_data(max_count=10_000):
    img_root = "//path/to/your/objarverse/views_release"
    data_root = "/path/to/your/gaussian-splatting/data/objaverse"
    output_root = "/path/to/your/gaussian-splatting/output/objaverse"
    save_root = "/path/to/your/gaussian-splatting/data/objaverse_transfer"

    transfer_json_path = "/path/to/your/gaussian-splatting/scripts/Objaverse/split/transfer.json"
    if os.path.exists(transfer_json_path):
        with open(transfer_json_path, "r") as file:
            transfer_json = json.load(file)
    else: transfer_json = []

    data_id_list = os.listdir(data_root)
    output_id_list = os.listdir(output_root)
    output_id_list.extend(transfer_json)

    ptr, transfer_id_list = 0, []
    data_id_list.sort()
    output_id_list.sort()

    for item_id in tqdm(output_id_list):
        if ptr >= len(data_id_list): break
        while data_id_list[ptr]<item_id:
            transfer_id_list.append(data_id_list[ptr])
            ptr += 1
            if ptr >= len(data_id_list): break
        if ptr >= len(data_id_list): break
        if data_id_list[ptr]==item_id:
            ptr += 1
    if ptr < len(data_id_list):
        transfer_id_list.extend(data_id_list[ptr:])
    print("data_id_list: ",len(data_id_list))
    print("output_id_list: ",len(output_id_list))
    print("transfer_id_list: ",len(transfer_id_list), len(data_id_list)-len(output_id_list))
    
    if "sh" in transfer_id_list: transfer_id_list.remove("sh")
    if "file_paths.json" in transfer_id_list: transfer_id_list.remove("file_paths.json")
    transfer_id_list = transfer_id_list[:max_count]

    transfer_json.extend(transfer_id_list)
    with open(transfer_json_path, "w") as file:
        json.dump(transfer_json, file, indent=4)

    os.makedirs(save_root, exist_ok=True)
    for item_id in tqdm(transfer_id_list, desc="copying "):
        save_item_root = os.path.join(save_root, item_id)
        train_img_root = os.path.join(save_item_root, "train")
        os.makedirs(save_item_root)
        os.makedirs(train_img_root)

        os.system("cp "+os.path.join(data_root, item_id, "pts.npy")+" "+os.path.join(save_root, item_id, "pts.npy"))
        os.system("cp "+os.path.join(data_root, item_id, "transforms_train.json")+" "+os.path.join(save_root, item_id, "transforms_train.json"))
        os.system("cp "+os.path.join(data_root, item_id, "transforms_test.json")+" "+os.path.join(save_root, item_id, "transforms_test.json"))

        for i in range(12):
            count_name = (3-len(str(i)))*"0"+str(i)+".png"
            os.system("cp "+os.path.join(img_root, item_id, count_name)+" "+os.path.join(train_img_root, count_name))

def copy_data_images():
    import multiprocessing
    item_root = "/path/to/your/gaussian-splatting/data/objaverse"
    save_root = "/path/to/your/gaussian-splatting/scripts/Objaverse/copy_tem"
    img_root = "//path/to/your/objarverse/views_release"

    os.makedirs(save_root, exist_ok=True)

    def copying_image(item_id_list, i):
        if i == 0: item_id_list = tqdm(item_id_list)
        for item_id in item_id_list:
            item_id = item_id.split(".")[0]
            
            from_item_root = os.path.join(item_root, item_id)
            to_item_root = os.path.join(save_root, item_id)
            os.makedirs(to_item_root, exist_ok=True)

            os.system("cp"+" "+os.path.join(from_item_root, "transforms_train.json")+" "+os.path.join(to_item_root, "transforms_train.json"))
            os.system("cp"+" "+os.path.join(from_item_root, "transforms_test.json")+" "+os.path.join(to_item_root, "transforms_test.json"))
            os.system("cp"+" "+os.path.join(from_item_root, "pts.npy")+" "+os.path.join(to_item_root, "pts.npy"))

            from_train_img_root, from_test_img_root = os.path.join(from_item_root, "train"), os.path.join(from_item_root, "test")
            to_train_img_root, to_test_img_root = os.path.join(to_item_root, "train"), os.path.join(to_item_root, "test")
            os.makedirs(to_train_img_root, exist_ok=True)
            os.makedirs(to_test_img_root, exist_ok=True)

            for img_name in os.listdir(from_train_img_root):
                from_img_path = os.path.join(img_root, item_id, img_name)
                to_img_path = os.path.join(to_train_img_root, img_name)
                os.system("cp"+" "+from_img_path+" "+to_img_path)
            for img_name in os.listdir(from_test_img_root):
                from_img_path = os.path.join(img_root, item_id, img_name)
                to_img_path = os.path.join(to_test_img_root, img_name)
                os.system("cp"+" "+from_img_path+" "+to_img_path)

            # if os.path.exists(os.path.join(save_root, item_id+".png")): continue
            # os.system("cp "+os.path.join(img_root, item_id, "000.png")+" "+os.path.join(save_root, item_id+".png"))
            # Image.open(os.path.join(img_root, item_id, "000.png")).convert("RGB").resize((224,224),resample=Image.Resampling.BILINEAR).save(os.path.join(save_root, item_id+".png"))

            # item_img_root = os.path.join(img_root, item_id)
            # save_img_root = os.path.join(save_root, item_id)
            # os.makedirs(save_img_root, exist_ok=True)

            # for i in range(12):
            #     ## 1
            #     count_name = (3-len(str(i)))*"0"+str(i)+".png"
            #     if os.path.exists(os.path.join(save_img_root, count_name)): continue
            #     os.system("cp "+os.path.join(item_img_root, count_name)+" "+os.path.join(save_img_root, count_name))

            #     ## 2
            #     # os.system("cp -r "+item_img_root+" "+save_img_root)

            #     ## 3
            #     # Image.open(os.path.join(item_img_root, count_name)).save(os.path.join(save_img_root, count_name))

    ## 1
    # all_item_id_list = os.listdir(copy_item_root)

    ## 2
    with open("/path/to/your/gaussian-splatting/scripts/Objaverse/to_move.json", "r") as file:
        all_item_id_list = json.load(file)
    
    # copying_image(all_item_id_list[:10], 0)
    # exit()

    split = 10
    gate = len(all_item_id_list)//split+1
    for i in range(split):
        p = multiprocessing.Process(target=copying_image, args=(all_item_id_list[gate*i:gate*(i+1)],i))
        p.start()

def add_pkl_attribute(root, prefix="root"):
    dataset_name = "abo"

    for pkl_name in tqdm(os.listdir(root),desc=prefix):
        pkl_path = os.path.join(root, pkl_name)
        if os.path.isdir(pkl_path): add_pkl_attribute(pkl_path, pkl_name)
        else:
            pkl_data = torch.load(pkl_path)
            if "dataset" in pkl_data.keys() and pkl_data["dataset"] == dataset_name: continue
            pkl_data["dataset"] = dataset_name
            torch.save(pkl_data, pkl_path)

def adding_file_paths():
    prompt_json_path = "/path/to/your/gaussian-splatting/data/objaverse/file_paths.json"
    with open(prompt_json_path, "r") as file:
        prompt_json = json.load(file)
    
    lvis_json_path = "//path/to/your/objarverse/lvis_items.json"
    with open(lvis_json_path, "r") as file:
        lvis_items = json.load(file)
    
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

    prompt_machine_path = "//path/to/your/objarverse/Cap3D_automated_Objaverse_full_no3Dword.csv"
    prompt_human_path = "//path/to/your/objarverse/Cap3D_human_Objaverse.pkl"

    prompt_machine_dict = get_item_prompt(prompt_machine_path)
    prompt_human_dict = get_item_prompt(prompt_human_path)

    object_paths_json_path = "//path/to/your/objarverse/object-paths.json"
    with open(object_paths_json_path, "r") as file:
        glb_object_paths = json.load(file)

    img_root = "//path/to/your/objarverse/views_release"
    glb_root = "//path/to/your/objarverse"
    npy_object_root = "//path/to/your/objarverse/openshape/objaverse-processed/merged_for_training_final/Objaverse/"

    for item_id in tqdm(lvis_items):
        if item_id not in prompt_json.keys():
            print(item_id)

            glb_path = os.path.join(glb_root, glb_object_paths[item_id])
            if os.path.exists(glb_path):
                pts_path = glb_path
            else:
                npy_subfix = glb_object_paths[item_id].repleace(".")[0][5:]+".npy"
                npy_path = os.path.join(npy_object_root, npy_subfix)

                assert os.path.exists(npy_path)
                pts_path = npy_path

            pts_path
            prompt_machine_dict[item_id]
            prompt_human_dict[item_id]

            prompt_json[item_id] = {
                "pts_path": pts_path,
                "img_path": os.path.join(img_root, item_id),
                "prompt_machine": prompt_machine_dict[item_id],
                "prompt_human": prompt_human_dict[item_id],
            }

    with open(prompt_json_path, "w") as file:
        json.dump(prompt_json, file, indent=4)

def get_update_json():

    lvis_json_path = "/path/to/your/gaussian-splatting/clip3/objaverse_all/sample_lvis.json"
    with open(lvis_json_path, "r") as file:
        lvis_file_paths = json.load(file)
    print("test", len(lvis_file_paths["test"]))

    label_json_path = "//path/to/your/objarverse/label.json"
    with open(label_json_path, "r") as file:
        label_json = json.load(file)

    new_test_file_paths, category_dict = [], {}
    for file_path in tqdm(lvis_file_paths["test"]):
        label = torch.load(file_path)["label"]
        if label not in category_dict.keys():
            category_dict[label] = [file_path]
        else: category_dict[label].append(file_path)

    count = 0
    new_test_key, test_key_mapping= {}, {}
    for key in category_dict:
        if len(category_dict[key])<50: continue
        new_test_key[key] = count
        count +=1 
        new_test_file_paths.extend(category_dict[key])
    print("count", count, len(new_test_file_paths))

    with open("//path/to/your/objarverse/label_50.json", "w") as file:
        json.dump(new_test_key, file, indent=4)

    # lvis_file_paths["test"] = new_test_file_paths
    # with open("/path/to/your/gaussian-splatting/clip3/objaverse_all/sample_lvis_50.json", "w") as file:
    #     json.dump(lvis_file_paths, file, indent=4)

if __name__ == "__main__":
    # find_objaverse()
    # count_objaverse_of_prompts()
    # dump_dataset_split()
    # recheck_dataset()

    # compare_objaverse_of_not_complete_model()
    # double_check_dataset()
    # transfer_data(10000)

    # adding_file_paths()

    # copy_data_images()
    add_pkl_attribute("/path/to/your/gaussian-splatting/clip3/ABO/objects")

    pass
