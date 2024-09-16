import os
import json
import torch
import random
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.utils.data.distributed import DistributedSampler

from base_dataset import BaseDataset

'''
2024-04-30 21:53:22 Epoch:[0]|[0|4500], Times:[ 11.446s| 0.520s, 0.354s], loss:3.2427, text_3d_loss:1.658670, image_3d_loss:1.584069
2024-04-30 21:53:26 Epoch:[0]|[10|4500], Times:[ 4.289s| 0.195s, 0.019s], loss:2.6351, text_3d_loss:1.326254, image_3d_loss:1.308880
2024-04-30 21:53:33 Epoch:[0]|[20|4500], Times:[ 6.725s| 0.306s, 0.020s], loss:2.3538, text_3d_loss:1.233769, image_3d_loss:1.120020
'''

class Mvimgnet(BaseDataset):
    # _xyz: [1024,3]
    # _features_dc: [1024,1,3]
    # _features_rest: [1024,15,3]
    # _opacity: [1024,1]
    # _scaling: [1024,3]
    # _rotation: [1024,4]

    def __init__(self,
        data_dir:str,
        data_split:str="file_paths.json",
        resolution:int=224,
        shuffle:bool=True,
        pc_prefix:str="point cloud of ",
        is_train:bool=True,
        using_channel:int=-1,
        load_extend=False,
        update_label=False,
        load_npy=False,
        **kwargs
        ):

        super().__init__(data_dir, data_split, resolution, shuffle, pc_prefix, is_train, using_channel, load_extend, load_npy)

        with open("/path/to/your/MVimgnet/InternLM_xcomposer2_caption.json", "r") as file:
            self.internlm_captions = json.load(file)
        
        self.update_label = update_label
        if update_label:
            label_update_path = "/path/to/your/MVimgnet/scripts/label_95.json"
            with open(label_update_path, "r") as file:
                self.update_label_json = json.load(file)
        
        self.load_npy = load_npy
        if load_npy:
            self.item_id_to_pts = {}
            self.pts_root = "/path/to/your/MVimgnet/mvi"
            for split in os.listdir(self.pts_root):
                for item_id in os.listdir(os.path.join(self.pts_root, split)):
                    self.item_id_to_pts[item_id] = os.path.join(self.pts_root, split, item_id, "sparse", "0", "points3D.ply") #???
        
        # self.file_paths = self.file_paths[::50]
    
    def gather_retrive_prompts(self):
        text_list, item_paths, text_item_dict = [], [], {}
        for index in range(len(self.file_paths)):
            pkl_path = self.file_paths[index]
            pkl_data = torch.load(pkl_path)

            text = self.pc_prefix + self.internlm_captions[pkl_data["name"]]
            
            # item_path
            item_path = pkl_data["item_path"]
            item_paths.append(item_path)

            text_item_dict[item_path]={
                "item_path": item_path,
                "text": text,
                "count": len(text_list),
            }
            text_list.append(text)
        
        for i,item_path in enumerate(item_paths):
            text = text_list[i]
            text_mask = (np.array(text_list) == text)
            text_mask[i] = False
            text_item_dict[item_path]["text_mask"] = text_mask

        return text_list, text_item_dict

    def _get_mvimgnet_caption(self, pkl_data):
        return self.pc_prefix + self.internlm_captions[pkl_data["name"]]

    def _get_mvimgnet_image_(self, pkl_data):
        img_root = pkl_data["img"].replace("/mvi/","/mvi_low/")
        exist_img_names = os.listdir(img_root)
        img_names = []
        for img_name in exist_img_names:
            if "bg_removed" not in img_name: img_names.append(img_name)
        img_path = os.path.join(img_root, img_names[np.random.randint(len(img_names))])
        image = Image.open(img_path).convert("RGB")

        return image
    
    def _getitem_3dgs_(self, pkl_data):
        #  51407 | 139282, R:[-1.92, -1.70, 0.25, 1.92, 6.35], G:[-1.92, -1.71, 0.14, 1.78, 6.50], B:[-1.92, -1.73, 0.02, 1.73, 6.31]

        if self.load_npy:
            item_id = pkl_data["name"]
            npy_path = self.item_id_to_pts[item_id]

            # xyzs, rgbs, _ = read_points3D_binary(npy_path)

            plydata = PlyData.read(npy_path)
            xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])),  axis=1)

            pkl_gaussians = np.concatenate((xyz, np.ones((xyz.shape[0], 11))*0.4), -1)
            pkl_gaussians = torch.from_numpy(pkl_gaussians).to(torch.float32)

            ran_sel = np.random.randint(0, pkl_gaussians.shape[0], 1_02400)
            pkl_gaussians = pkl_gaussians[ran_sel]
        
        else:
            pkl_gaussians = torch.from_numpy(pkl_data["3dgs"])
            pkl_gaussians = self.expand_3dgs(pkl_gaussians)

            pkl_gaussians[:,3:] = torch.tanh(pkl_gaussians[:,3:])
            pkl_gaussians[:,6:] = 0.4

            x, y,z = pkl_gaussians[:,0:1].clone(), pkl_gaussians[:,1:2].clone(), pkl_gaussians[:,2:3].clone()
            pkl_gaussians[:,0:1] = x
            pkl_gaussians[:,1:2] = z
            pkl_gaussians[:,2:3] = -y

            if self.using_channel>0:
                pkl_gaussians = pkl_gaussians[:,:self.using_channel]

        return pkl_gaussians.transpose(0,1)

    def _getitem_extra_(self, pkl_data, meta_data):
        if "label_count" in pkl_data.keys():
            meta_data["label_count"] = int(pkl_data["label_count"])
            if self.update_label:
                meta_data["label_count"] = int(self.update_label_json[pkl_data["label"]])
        else: meta_data["label_count"] = 0
        meta_data["valid"] = (meta_data['text'] != "")
        return meta_data

    def __getitem__(self, index):
        '''
        name:                           <numpy.str_>
        img:            [325, 115, 3]   <numpy.ndarray>
        className2D:                    <numpy.str_>
        3dgs:           [13277, 14]     <numpy.ndarray>
        className3D:                    <numpy.str_> (old [1, ] | new [])
        '''
        pkl_path = self.file_paths[index]
        pkl_data = torch.load(pkl_path)
        meta_data = {}

        meta_data["image"] = self._getitem_image_(pkl_data)
        meta_data["gaussian"] = self._getitem_3dgs_(pkl_data)
        meta_data["text"] = self._getitem_caption_(pkl_data)

        meta_data["item_path"] = pkl_data["item_path"]

        meta_data = self._getitem_extra_(pkl_data, meta_data)

        return meta_data