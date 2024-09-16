import os
import json
import torch
import numpy as np
from tqdm import tqdm
import random
# np.set_printoptions(precision=3, suppress=True)

item_id = "3d56bea8818747ed8831fdb085b07c90"
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

print(prompt_machine_dict[item_id])
print(prompt_human_dict[item_id])