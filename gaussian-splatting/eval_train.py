import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import torchvision.transforms.functional as tf

"CUDA_VISIBLE_DEVICES=0 python render.py -m output/NYU0001 --skip_test"

selected_items = []
root = "/path/to/your/gaussian-splatting/data/SUNRGBD"
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

psnrs_list = [[],[],[]]
root = "/path/to/your/gaussian-splatting/output/SUNRGBD"
for item in tqdm(selected_items):
    if os.path.exists(os.path.join(root, item, "train")): continue
    line = "CUDA_VISIBLE_DEVICES=0 python render.py -m output/SUNRGBD/"+item+" --skip_test"
    print(line)
    exit()
    os.system(line)

for item in tqdm(selected_items):
    item_root = os.path.join(root, item, "train", "ours_500")
    gt_img = Image.open(os.path.join(item_root,"gt","00000.png"))
    redner_img = Image.open(os.path.join(item_root,"renders","00000.png"))

    gt_img = tf.to_tensor(gt_img).unsqueeze(0)[:, :3, :, :].cuda()
    redner_img = tf.to_tensor(redner_img).unsqueeze(0)[:, :3, :, :].cuda()

    print(psnr(redner_img, gt_img), 
          ssim(redner_img, gt_img), 
          lpips(redner_img, gt_img, net_type='vgg'))
    
    exit()
