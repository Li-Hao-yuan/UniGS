import os
from tqdm import tqdm
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image
from sunrgbd import SUN

# dataset item
meta_file_path = "//path/to/your/SUNRGBD/SUNRGBDMeta3DBB_v2.mat"
meta_file_2D_path = "//path/to/your/SUNRGBD/SUNRGBDMeta2DBB_v2.mat"
root = "//path/to/your/SUNRGBD"
sun_data = SUN(meta_file_path, meta_file_2D_path, root, using_fbx=True)

write_test_set_img = False

# get item id 
kv1Index=sun_data.getSensorDataId('kv1')
kv2Index=sun_data.getSensorDataId('kv2')
realsenseIndex=sun_data.getSensorDataId('realsense')
xtionIndex=sun_data.getSensorDataId('xtion')
allIndex = [*kv1Index, *kv2Index, *realsenseIndex, *xtionIndex] # 10335
# allIndex = [*realsenseIndex] # 10335

for item_id in tqdm(allIndex):
    # get save path
    save_root = "/path/to/your/gaussian-splatting/data"
    sequenceName=sun_data.dataSet[item_id][0][0]
    save_item_path = os.path.join(save_root, sequenceName)

    # if os.path.exists(save_item_path): continue

    # get image
    img,depth,segl,segi=sun_data.getImg(item_id)
    points3d, _ = sun_data.load3dPoints(item_id) # [x,y,z]
    h, w = img.shape[0], img.shape[1]

    mask = np.sum(np.isnan(points3d),axis=-1)
    points3d = points3d[mask==0]

    mean_x, mean_y, mean_z = np.mean(points3d[:,0]), np.mean(points3d[:,1]), np.mean(points3d[:,2])
    points3d -= [mean_x, mean_y, mean_z]

    # save path
    os.makedirs(save_item_path, exist_ok=True)
    pts_save_path = os.path.join(save_item_path, "pts.npy")
    np.save(pts_save_path, points3d)

    # get intrinsics and extrinsics
    item_path = os.path.join(root, sequenceName)
    if os.path.exists(os.path.join(item_path, "fullres", "intrinsics.txt")):
        intrinsics = np.loadtxt(os.path.join(item_path, "fullres", "intrinsics.txt"))
    else: intrinsics = np.loadtxt(os.path.join(item_path, "intrinsics.txt"))
    intrinsics = np.reshape(intrinsics,(-1))

    extrinsics_root = os.path.join(item_path, "extrinsics")
    extrinsics_name = os.listdir(extrinsics_root)[-1]
    extrinsics = np.loadtxt(os.path.join(extrinsics_root, extrinsics_name))

    transform_matrix = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
    extrinsics[:3,:3] = np.matmul(extrinsics[:3,:3], transform_matrix)

    extrinsics = extrinsics.tolist()
    extrinsics.append([0,0,0,1])

    # offset
    extrinsics[0][3] -= mean_x
    extrinsics[1][3] -= mean_y
    extrinsics[2][3] -= mean_z
    with open(os.path.join(save_item_path, "offset.txt"), "w") as file:
        file.write(str(mean_x)+","+str(mean_y)+","+str(mean_z))

    # create json
    camera_angle_x = 2*np.arctan(w/2/intrinsics[0])
    empty_image = np.zeros((h,w,3))
    template_json_path = "//path/to/your/SUNRGBD/template.json"
    with open(template_json_path, "r") as file:
        json_content = json.load(file)
    json_content["camera_angle_x"] = camera_angle_x

    # test set
    with open(os.path.join(save_item_path,"transforms_test.json"), "w") as file:
        json.dump(json_content, file, indent=4)
        
    if write_test_set_img:
        test_root = os.path.join(save_item_path, "test")
        os.makedirs(test_root, exist_ok=True)
        img_name = json_content["frames"][0]["file_path"].split("/")[-1]+".png"
        first_img_path = os.path.join(test_root, img_name)
        plt.imsave(first_img_path, empty_image)
        for item in json_content["frames"][1:]:
            img_name = item["file_path"].split("/")[-1]+".png"
            os.symlink(first_img_path, os.path.join(test_root, img_name))

    # train set
    train_json_content = json_content.copy()
    train_json_content["frames"] = [train_json_content["frames"][0]]
    train_json_content["frames"][0]["file_path"] = train_json_content["frames"][0]["file_path"].replace("test","train")
    train_json_content["frames"][0]["transform_matrix"] = extrinsics
    with open(os.path.join(save_item_path,"transforms_train.json"), "w") as file:
        json.dump(train_json_content, file, indent=4)
    train_root = os.path.join(save_item_path, "train")
    os.makedirs(train_root, exist_ok=True)
    img_name =  train_json_content["frames"][0]["file_path"].split("/")[-1]+".png"
    plt.imsave(os.path.join(train_root,img_name), img)

