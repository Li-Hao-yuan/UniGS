import os
import numpy as np
from PIL import Image
import json
import imageio
from tqdm import tqdm

R = 3 # 3
W = 480

fovx = 60
camera_angle_x = np.deg2rad(fovx)
focal = (0.5*W)/np.tan(camera_angle_x/2)

focal = 560

transform_matrix = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]])

def transfer_pose(azimuth, elevation):
    w2c_pose = np.array([
        [-np.sin(azimuth), np.cos(azimuth), 0, 0],
        [np.sin(elevation)*np.cos(azimuth), np.sin(elevation)*np.sin(azimuth), -np.cos(elevation), 0],
        [-np.cos(elevation)*np.cos(azimuth), -np.cos(elevation)*np.sin(azimuth), -np.sin(elevation), R],
        [0,0,0,1]
    ])
    c2w_pose = np.linalg.inv(w2c_pose)
    c2w_pose[:3,:3] = np.matmul(c2w_pose[:3,:3], transform_matrix) # ABO

    return c2w_pose.tolist()

def transfer_test(
        output_json_path = "/path/to/your/gaussian-splatting/scripts/template/transforms_test.json",
        template_json_path = "/path/to/your/gaussian-splatting/scripts/utils/template/transforms_test_save.json",
):
    with open(template_json_path, "r") as file:
        content = json.load(file)

    for i in range(2):
        for j in range(36):
            azimuth = (j*10)/180*np.pi
            elevation = (-30+i*60)/180*np.pi
            content["frames"][i*36+j]["transform_matrix"] = transfer_pose(azimuth, elevation)
    content["camera_angle_x"] = np.arctan( 0.5 * W / focal) * 2

    with open(output_json_path, "w") as file:
        json.dump(content, file, indent=4)


def transfer_train(
        input_json_path = "/path/to/your/gaussian-splatting/scripts/template/transforms_train_save.json",
        output_json_path  = "/path/to/your/gaussian-splatting/scripts/template/transforms_train.json",
):
    with open(input_json_path, "r") as file:
        content = json.load(file)
    
    azimuths = [30, 90, 150, 210, 270, 330]
    elevations = [30, -20, 30, -20, 30, -20]

    for i in range(6):
        azimuth = azimuths[i]/180*np.pi
        elevation = elevations[i]/180*np.pi
        content["frames"][i]["transform_matrix"] = transfer_pose(azimuth, elevation)
    content["camera_angle_x"] = np.arctan( 0.5 * W / focal) * 2

    with open(output_json_path, "w") as file:
        json.dump(content, file, indent=4)

def transfer_template(
        root = "/path/to/your/gaussian-splatting/data/objaverse/dfdf414b3e3146acbe4d26fdf29257d5",
):
    template_img_path = os.path.join(root,"train", "000.png")

    for camera in ["down", "up"]:
        first_img_path = None
        for i in range(36):
            count_name = (3-len(str(i)))*"0"+str(i)
            save_img_path = os.path.join(root, "test", camera+"_"+count_name+".png")
            if first_img_path is None:
                img = np.array(Image.open(template_img_path).convert("RGB"))
                img -= img
                Image.fromarray(img).save(save_img_path)
                first_img_path = save_img_path
            else: os.symlink(first_img_path, save_img_path)


if __name__ == "__main__":
    # transfer_train()

    root = "/path/to/your/gaussian-splatting/data/objaverse_test/0000954cf268495ca39760d3d8e11862"
    transfer_test(
        root+"/transforms_test.json"
    )
    transfer_template(
        root
    )