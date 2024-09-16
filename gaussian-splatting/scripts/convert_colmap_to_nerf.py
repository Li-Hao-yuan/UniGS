import os, sys
root_dir = os.path.join("/path/to/your/gaussian-splatting")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import json
import numpy as np
from PIL import Image
from typing import NamedTuple
from pathlib import Path
from shutil import rmtree
import math
from plyfile import PlyData, PlyElement

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapSceneInfo(path, images="images", llffhold=8):

    path = path.replace("clip3D/gaussian-splatting/", "")

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")

        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    xyz, _, _ = read_points3D_binary(bin_path)

    # bin_path = os.path.join(path, "sparse/0/points3D.ply")
    # plydata = PlyData.read(bin_path)
    # xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
    #                 np.asarray(plydata.elements[0]["y"]),
    #                 np.asarray(plydata.elements[0]["z"])),  axis=1)

    offset = np.array([np.mean(xyz[:,0]), np.mean(xyz[:,1]), np.mean(xyz[:,2])])
    offset = [0,0,0]

    xyz[:,0] -= offset[0]
    xyz[:,1] -= offset[1]
    xyz[:,2] -= offset[2]

    return train_cam_infos, test_cam_infos, offset, xyz

def cam_infos_to_json(cam_infos, split, offset):
    image_paths = []
    camera_pose_json = {
        "camera_angle_x": cam_infos[0].FovX,
        "camera_angle_y": cam_infos[0].FovY,
        "frames":[]
    }
    transform_matrix = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]])
    for cam_info in cam_infos:
        camera_pose = np.concatenate((cam_info.R, [cam_info.T]), axis=0).T
        camera_pose = np.concatenate((camera_pose, [[0,0,0,1]]), axis=0)
        camera_pose = np.linalg.inv(camera_pose)
        camera_pose[0,-1] -= offset[0]
        camera_pose[1,-1] -= offset[1]
        camera_pose[2,-1] -= offset[2]
        camera_pose[:3,:3] = np.matmul(camera_pose[:3,:3], transform_matrix)
        camera_pose_json["frames"].append({
            "file_path": split+"/"+cam_info.image_name,
            "transform_matrix": camera_pose.tolist(),
        })
        image_paths.append(cam_info.image_path)
    return camera_pose_json, image_paths



def transfer_colmap_to_nerf(item_data_dir, use_mask=False, white_bg=False, template_test=False, llffhold=8):
    train_cam_infos, test_cam_infos, offset, xyz = readColmapSceneInfo(item_data_dir, llffhold=llffhold)
    train_json, train_image_paths = cam_infos_to_json(train_cam_infos, "train", offset)
    test_json, test_image_paths = cam_infos_to_json(test_cam_infos, "test", offset)

    if os.path.exists(os.path.join(item_data_dir, "train")): rmtree(os.path.join(item_data_dir, "train"))
    os.makedirs(os.path.join(item_data_dir, "train"))
    for img_path in train_image_paths: 
        image_name = img_path.split("/")[-1]
        if not use_mask:
            os.symlink(img_path, os.path.join(item_data_dir, "train", image_name.split(".")[0]+".png"))
        else:
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(os.path.join(item_data_dir, "masks", image_name)))
            if len(mask.shape)>2: mask = mask[:,:,3:]
            mask = mask/np.max(mask)
            img = img * mask + int(white_bg) * np.ones_like(img) * (1-mask) * 255
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(item_data_dir, "train", image_name.split(".")[0]+".png"))

    with open(os.path.join(item_data_dir, "transforms_train.json"), "w") as file:
        json.dump(train_json, file, indent=4)
    np.save(os.path.join(item_data_dir, "convert.npy"), xyz)

    if os.path.exists(os.path.join(item_data_dir, "test")): rmtree(os.path.join(item_data_dir, "test"))
    os.makedirs(os.path.join(item_data_dir, "test"))
    if template_test:
        pass
    else:
        for img_path in test_image_paths: 
            image_name = img_path.split("/")[-1]
            if not use_mask:
                os.symlink(img_path, os.path.join(item_data_dir, "test", image_name.split(".")[0]+".png"))
            else:
                img = np.array(Image.open(img_path))
                mask = np.array(Image.open(os.path.join(item_data_dir, "masks", image_name)))
                if len(mask.shape)>2: mask = mask[:,:,3:]
                mask = mask/np.max(mask)
                img = img * mask + int(white_bg) * np.ones_like(img) * (1-mask) * 255
                Image.fromarray(img.astype(np.uint8)).save(os.path.join(item_data_dir, "test", image_name.split(".")[0]+".png"))

        with open(os.path.join(item_data_dir, "transforms_test.json"), "w") as file:
            json.dump(test_json, file, indent=4)

item_data_dir = "clip3D/gaussian-splatting/data/1d00a186"
transfer_colmap_to_nerf(item_data_dir, False)