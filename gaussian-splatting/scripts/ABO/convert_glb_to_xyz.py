import os
from tqdm import tqdm
import aspose.threed as a3d
import open3d as o3d
from plyfile import PlyData, PlyElement
import numpy as np

glb_root = "//path/to/your/ABO/3dmodels/original"
xyz_root = "//path/to/your/ABO/3dmodels/xyz"
ply_root = "//path/to/your/ABO/3dmodels/ply"
reference_root = "//path/to/your/ABO/render_images_256"

# seach for item type
item_type_dict = {}
for data_type in os.listdir(reference_root):
    for item_id in os.listdir( os.path.join(reference_root, data_type) ):
        item_type_dict[item_id] = data_type

# transfer
transfer_item_count = 0
for split in os.listdir(glb_root):
    split_root = os.path.join(glb_root, split)

    for item_name in tqdm(os.listdir(split_root)):
        item_id = item_name.split(".")[0]
        if item_id not in item_type_dict.keys(): continue

        item_path = os.path.join(split_root, item_name)

        to_save_path = os.path.join(ply_root, item_type_dict[item_id])
        os.makedirs(to_save_path, exist_ok=True)

        item_to_save_path = os.path.join(to_save_path, item_id+".ply")
        if not os.path.exists(item_to_save_path):
            glb_item = o3d.io.read_triangle_mesh(item_path)
            glb_pts = glb_item.sample_points_uniformly(number_of_points=100_000)
            o3d.io.write_point_cloud(item_to_save_path, glb_pts)
            transfer_item_count += 1

print("PLY save items : ", transfer_item_count)

# xyz
transfer_item_count = 0
for data_type in tqdm(os.listdir(ply_root)):
    ply_type_root = os.path.join(ply_root, data_type)

    for item_name in os.listdir(ply_type_root):
        item_id = item_name.split(".")[0]

        ply_item_path = os.path.join(ply_type_root, item_name)

        to_save_path = os.path.join(xyz_root, data_type)
        os.makedirs(to_save_path, exist_ok=True)

        item_to_save_path = os.path.join(to_save_path, item_id+".npy")
        if not os.path.exists(item_to_save_path):
            plydata = PlyData.read(ply_item_path)
            xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                                    np.asarray(plydata.elements[0]["y"]),
                                    np.asarray(plydata.elements[0]["z"])),  axis=1)
            np.save(item_to_save_path, xyz)
            transfer_item_count += 1 
print("XYZ save items : ", transfer_item_count)