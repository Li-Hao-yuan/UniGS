import os
import json

# root = "/path/to/your/gaussian-splatting/output/objaverse" # 29.32
# root = "/path/to/your/gaussian-splatting/output/SUNRGBD" # 31.19
root = "/path/to/your/gaussian-splatting/output/mvimgnet_500" # 30.56

def scan_one_folder(folder_path):
    training_result_list, training_result_dict = [], {}
    for item_id in os.listdir(folder_path):
        if not os.path.isdir(os.path.join(folder_path, item_id)): continue
        json_path = os.path.join(folder_path, item_id, "training_results.json")
        if not os.path.exists(json_path): 
            result_list, result_dict = scan_one_folder(os.path.join(folder_path, item_id))
            training_result_list.extend(result_list)
            training_result_dict.update(result_dict)
        else:
            with open(json_path, "r") as file:
                psnr = json.load(file)["train"]["psnr"]
                training_result_list.append(psnr)
                training_result_dict[item_id] = psnr
            return training_result_list, training_result_dict
    return training_result_list, training_result_dict

training_result_list, training_result_dict = scan_one_folder(root)
print( sum(training_result_list)/len(training_result_list) )



'''
29.322425926985698
'''