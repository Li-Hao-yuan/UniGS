import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import torch
from plyfile import PlyData, PlyElement
import time
from PIL import Image
import h5py
import pathlib
import json
from sunrgbd import SUN

# nohup python /path/to/your/gaussian-splatting/scripts/convert_3DGS_to_pkl.py > "/path/to/your/gaussian-splatting/scripts/running.log" 2>& 1&

def get_file(file_path, pattern="*"):
    all_file = []
    files = pathlib.Path(file_path).rglob(pattern)
    for file in files:
        if pathlib.Path.is_file(file):
            all_file.append(file)
    return all_file

def load_ply(path, offset, max_sh_degree=0):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"])+offset[0],
                    np.asarray(plydata.elements[0]["y"])+offset[1],
                    np.asarray(plydata.elements[0]["z"])+offset[2]),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # _xyz: [1024,3]
    # _features_dc: [1024,1,3]
    # _features_rest: [1024,15,sh_dgree]
    # _opacity: [1024,1]
    # _scaling: [1024,3]
    # _rotation: [1024,4]

    _xyz = torch.tensor(xyz, dtype=torch.float)
    num_pts = _xyz.shape[0]
    _features_dc = torch.tensor(features_dc, dtype=torch.float).transpose(1, 2).contiguous().reshape(num_pts, -1) # 3,1 -> 1,3 -> 3
    _features_rest = torch.tensor(features_extra, dtype=torch.float).transpose(1, 2).contiguous().reshape(num_pts, -1)
    _opacity = torch.tensor(opacities, dtype=torch.float)
    _scaling = torch.tensor(scales, dtype=torch.float)
    _rotation = torch.tensor(rots, dtype=torch.float)

    pts_feature = torch.cat(
        (_xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation), dim=1
    )

    return pts_feature

def save_ply(data, save_path):

    # _xyz: [1024,3]
    # _features_dc: [1024,1,3]
    # _features_rest: [1024,15,3]
    # _opacity: [1024,1]
    # _scaling: [1024,3]
    # _rotation: [1024,4]

    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l
    
    xyz = data[:,:3].data.numpy()
    f_dc = data[:,3:6].reshape((-1,1,3)).transpose(1, 2).flatten(start_dim=1).data.numpy() # 1,3 -> 3,1 -> 3
    f_rest = data[:,6:51].reshape((-1,15,3)).transpose(1, 2).flatten(start_dim=1).data.numpy()
    opacities = data[:,51:52].data.numpy()
    scale = data[:,52:55].data.numpy()
    rotation = data[:,55:59].data.numpy()
    normals = np.zeros_like(xyz)

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(save_path)

# dataset item
meta_file_path = "//path/to/your/SUNRGBD/SUNRGBDMeta3DBB_v2.mat"
meta_file_2D_path = "//path/to/your/SUNRGBD/SUNRGBDMeta2DBB_v2.mat"
root = "//path/to/your/SUNRGBD"
sun_data = SUN(meta_file_path, meta_file_2D_path, root, using_fbx=True)

# get item id 
kv1Index=sun_data.getSensorDataId('kv1')
kv2Index=sun_data.getSensorDataId('kv2')
realsenseIndex=sun_data.getSensorDataId('realsense')
xtionIndex=sun_data.getSensorDataId('xtion')
allIndex = [*kv1Index, *kv2Index, *realsenseIndex, *xtionIndex] # 10335

root = "/path/to/your/gaussian-splatting/data"
output_data_root = "/path/to/your/gaussian-splatting/output"
save_data_root = "/path/to/your/gaussian-splatting/clip3/raw_scene"
os.makedirs(save_data_root, exist_ok=True)

# split json
data_split_json, data_to_split = sun_data.get_split_json()
with open(os.path.join(save_data_root, "split.json"), "w") as file:
    json.dump(data_split_json, file, indent=4)

# seg
seg_mat_path = "//path/to/your/SUNRGBD/SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat"
SUNRGBD2Dseg = h5py.File(seg_mat_path, mode='r', libver='latest')
seglabel = SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

debug = False
# allIndex = [0]
# allIndex = allIndex[233:]
# allIndex = allIndex[10000:]

scenes_count = 0
scene_save_interval = 1000

FLUSH = True
scenes_data = []
all_objects = 0
useful_objects = 0
outside_37seglist_count = 0
begin_time = time.perf_counter()

for item_id in tqdm(allIndex):
    # root path
    sequenceName=sun_data.dataSet[item_id][0][0]
    ori_data_path = os.path.join(root, sequenceName)
    guassian_output_path = os.path.join(output_data_root, sequenceName)
    save_item_path = os.path.join(save_data_root, sequenceName)

    # read 3d feature
    offset_path = os.path.join(ori_data_path, "offset.txt")
    offset = []
    with open(offset_path, "r") as file:
        xyz_offset = file.readline().replace(" ","").split(",")
        for offset_axis in xyz_offset: offset.append(float(offset_axis))
    point_cloud_path = os.path.join(guassian_output_path, "point_cloud", "iteration_500", "point_cloud.ply")
    ply_feature = load_ply(point_cloud_path, offset)

    # read bbox
    cornerList3D, classNameList3D =sun_data.getCornerList(item_id) # classname maybe not the same
    cornerList2D, classNameList2D =sun_data.getCornerList2D(item_id,ensure_3d=False) # 实际上有些has3dbox是不对的
    img = np.array(sun_data.getImg(item_id,only_img=True))
    h, w = img.shape[0], img.shape[1]

    # color as index
    color_index_1 = np.linspace(0,0.99,100).reshape((100,1,1))*np.ones((1,100,100))
    color_index_2 = np.linspace(0,0.99,100).reshape((1,100,1))*np.ones((1,1,100)).repeat(100,0)
    color_index_3 = np.linspace(0,0.99,100).reshape((1,1,100)).repeat(100,1).repeat(100,0)
    color_indexs = np.concatenate((np.expand_dims(color_index_1, -1), np.expand_dims(color_index_2, -1), np.expand_dims(color_index_3, -1)),axis=-1)
    color_indexs = np.reshape(color_indexs, (-1, 3)) # [1000000, 3]

    this_scene_data = {
        "name":sequenceName,
        "split":data_to_split[sequenceName],
        "label_count":[],
        "img":[],
        "className2D":[],
        "3dgs":[],
        "className3D":[],
        'bbox2d':[],
        "bbox3d":[],
        }
    if debug:
        img_path = os.path.join(save_item_path, "img")
        pc_path = os.path.join(save_item_path, "point_cloud")

        os.makedirs(save_item_path, exist_ok=True)
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(pc_path, exist_ok=True)

        selected_items = []
        for i,corner in enumerate(cornerList2D):
            #######################
            if corner[2] < 10 or corner[3]<10: continue
            if i >= len(cornerList3D): continue
            if np.max(cornerList3D[i][:,0]) - np.min(cornerList3D[i][:,0]) < 1e-8: continue
            if np.max(cornerList3D[i][:,1]) - np.min(cornerList3D[i][:,1]) < 1e-8: continue
            if np.max(cornerList3D[i][:,2]) - np.min(cornerList3D[i][:,2]) < 1e-8: continue
            #######################

            label_img = np.array(SUNRGBD2Dseg[seglabel[item_id][0]][:].transpose(1, 0)).astype(np.uint8)
            centeroid = [int(corner[1]+corner[3]/2), int(corner[0]+corner[2]/2)]
            centeroid[0] = h-1 if centeroid[0] >= h else centeroid[0]
            centeroid[1] = w-1 if centeroid[1] >= w else centeroid[1]
            label_count = label_img[centeroid[0], centeroid[1]]

            selected_items.append(i)
            corner = np.array(corner).astype(np.uint16)

            crop_img = img[corner[1]:corner[1]+corner[3], corner[0]:corner[0]+corner[2]]
            plt.imsave(os.path.join(img_path, classNameList2D[i]+"_"+str(i)+"_"+str(label_count)+".png"), crop_img)

        points3d = ply_feature[:,:3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points3d)
        pcd.paint_uniform_color([1., 1., 0.])
        np.asarray(pcd.colors)[:,:] = color_indexs[:points3d.shape[0]]

        for i,corner in enumerate(cornerList3D):
            if i not in selected_items: continue
            '''
            garbage_bin (3702, 3)
            chair (7554, 3)
            garbage_bin (4645, 3)
            chair (1460, 3)
            side_table (155, 3)
            counter (35, 3)
            sink (598, 3)
            cabinet (8232, 3)
            counter (11923, 3)
            microwave (2632, 3)
            papertoweldispenser (1350, 3)
            fire_place (1155, 3)
            '''

            bounding_polygon = np.array(corner).astype("float64")
            bbox = o3d.utility.Vector3dVector(corner)
            oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(bbox)
            oriented_bounding_box.color = (1,0,0)

            point_cloud_crop = pcd.crop(oriented_bounding_box)
            colors = np.asarray(point_cloud_crop.colors)
            indexs = (colors[:,0]*10**6 + colors[:,1]*10**4 + colors[:,2]*10**2).astype(np.int64)
            sample_ply = ply_feature[indexs] # tensor
            sample_ply[:,:3] -= torch.tensor(offset)
            save_ply(sample_ply, os.path.join(pc_path, classNameList3D[i]+"."+str(i)+".ply"))

        print("exiting | item_id : [",item_id, "], sequenceName : ", sequenceName)
        exit()
    else:
        selected_items = []
        for i,corner in enumerate(cornerList2D):
            all_objects += 1
            corner = np.array(corner).astype(np.int32)
            crop_img = img[corner[1]:corner[1]+corner[3], corner[0]:corner[0]+corner[2]]

            ####################### 筛选条件
            if corner[2] < 10 or corner[3]<10: continue
            if i >= len(cornerList3D): continue
            if np.max(cornerList3D[i][:,0]) - np.min(cornerList3D[i][:,0]) < 1e-8: continue
            if np.max(cornerList3D[i][:,1]) - np.min(cornerList3D[i][:,1]) < 1e-8: continue
            if np.max(cornerList3D[i][:,2]) - np.min(cornerList3D[i][:,2]) < 1e-8: continue
            #######################

            label_img = np.array(SUNRGBD2Dseg[seglabel[item_id][0]][:].transpose(1, 0)).astype(np.uint8)
            centeroid = [int(corner[1]+corner[3]/2), int(corner[0]+corner[2]/2)]
            centeroid[0] = h-1 if centeroid[0] >= h else centeroid[0]
            centeroid[1] = w-1 if centeroid[1] >= w else centeroid[1]
            label_count = label_img[centeroid[0], centeroid[1]]

            selected_items.append(i)
            useful_objects += 1
            if label_count == 0: outside_37seglist_count += 1

            this_scene_data["img"].append(crop_img)
            this_scene_data["label_count"].append(label_count)
            this_scene_data["className2D"].append(classNameList2D[i])
            this_scene_data["bbox2d"].append(corner)

        points3d = ply_feature[:,:3]
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points3d)
        pcd.paint_uniform_color([1., 1., 0.])
        np.asarray(pcd.colors)[:,:] = color_indexs[:points3d.shape[0]]

        for i,corner in enumerate(cornerList3D):
            if i not in selected_items: continue
            
            bounding_polygon = np.array(corner).astype("float64")

            bbox = o3d.utility.Vector3dVector(corner)
            oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(bbox)
            point_cloud_crop = pcd.crop(oriented_bounding_box)

            colors = np.asarray(point_cloud_crop.colors)

            indexs = (colors[:,0]*10**6 + colors[:,1]*10**4 + colors[:,2]*10**2).astype(np.int64)
            sample_ply = ply_feature[indexs]

            this_scene_data["3dgs"].append(sample_ply.data.numpy())
            this_scene_data["className3D"].append(classNameList3D[i])
            this_scene_data["bbox3d"].append(corner)
        
        this_scene_data["img"] = this_scene_data["img"][:len(cornerList3D)]
        this_scene_data["className2D"] = this_scene_data["className2D"][:len(cornerList3D)]
        this_scene_data["3dgs"] = this_scene_data["3dgs"]
        this_scene_data["className3D"] = this_scene_data["className3D"]
        
        scenes_count += 1
        scenes_data.append(this_scene_data)
        if scenes_count % scene_save_interval == 0 or scenes_count == len(allIndex):
            if scenes_count == len(allIndex): scenes_count += 1000
            pkl_name = "SUNRGBD_"+str(int(scenes_count//scene_save_interval))+".pkl"
            print("\nSaving scene data : %s"%(pkl_name), end="", flush=FLUSH)
            torch.save(scenes_data, os.path.join(save_data_root, pkl_name))
            del scenes_data
            scenes_data = []


    if debug: exit()

# save data
time_cost = time.perf_counter() - begin_time 
print("Collect %d object from %d scenes"%(useful_objects, len(allIndex)), flush=FLUSH)
print("Time costing | All : %.4fs, Ave[scene]: %.4fs, Ave[object]: %.4fs"%(time_cost, time_cost/len(allIndex), time_cost/useful_objects ), flush=FLUSH)
print("Objects: All: %d, loading: %d, outside: %d"%(all_objects, useful_objects, outside_37seglist_count))
print("save ok!", flush=FLUSH)

# Objects: All: 80220, loading: 64347, outside: 22156 (42191)