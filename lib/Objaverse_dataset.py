import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.utils.data.distributed import DistributedSampler

from base_dataset import BaseDataset

'''
2024-04-30 21:53:22 Epoch:[0]|[0|4500], Times:[ 11.446s| 0.520s, 0.354s], loss:3.2427, text_3d_loss:1.658670, image_3d_loss:1.584069
2024-04-30 21:53:26 Epoch:[0]|[10|4500], Times:[ 4.289s| 0.195s, 0.019s], loss:2.6351, text_3d_loss:1.326254, image_3d_loss:1.308880
2024-04-30 21:53:33 Epoch:[0]|[20|4500], Times:[ 6.725s| 0.306s, 0.020s], loss:2.3538, text_3d_loss:1.233769, image_3d_loss:1.120020
'''

class Objaverse(BaseDataset):
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
        load_blip2=False,
        load_interlm=False,
        update_label=False,
        load_npy=False,
        **kwargs
        ):

        super().__init__(data_dir, data_split, resolution, shuffle, pc_prefix, is_train, using_channel, load_extend, load_npy)

        self.load_blip2 = load_blip2
        if load_blip2:
            print("Dataset load Blip2 as caption!")
            blip2_caption_path = "/path/to/your/objarverse/blip2_caption.json"
            with open(blip2_caption_path, "r") as file:
                self.blip2_caption = json.load(file)
            
        self.load_interlm = load_interlm
        if load_interlm:
            print("Dataset load InterLM as caption!")
            interlm_caption_path = "/path/to/your/objarverse/InternLM_xcomposer2_caption.json"
            with open(interlm_caption_path, "r") as file:
                self.interlm_caption = json.load(file)

        self.load_npy = load_npy
        if load_npy:
            self.item_id_to_pts = {}
            self.pts_root = "/path/to/your/gaussian-splatting/data/objaverse"
            for item_id in os.listdir(self.pts_root):
                self.item_id_to_pts[item_id] = os.path.join(self.pts_root, item_id, "pts.npy")

        if load_extend:
            print("Load extra dataset as supplement!")

            if load_blip2:
                sun_blip_caption_path = "/path/to/your/gaussian-splatting/unigs/sunrgbd_all/blip2_caption.json"
                with open(sun_blip_caption_path, "r") as file:
                    self.sun_blip_captions = json.load(file)
        
        self.update_label = update_label
        if update_label:
            label_update_path = "/path/to/your/objarverse/label_50.json"
            with open(label_update_path, "r") as file:
                self.update_label_json = json.load(file)

        # self.file_paths = self.file_paths[::100]
    
    def gather_retrive_prompts(self):
        text_list, item_paths, text_item_dict = [], [], {}
        for index in range(len(self.file_paths)):
            pkl_path = self.file_paths[index]
            pkl_data = torch.load(pkl_path)

            # text
            if self.load_interlm:
                text = self.pc_prefix + self.interlm_caption[pkl_data["name"]]
            elif "human_prompt" in pkl_data.keys():
                text = self.pc_prefix + pkl_data["human_prompt"][0]
            elif "machine_prompt" in pkl_data.keys():
                if isinstance(pkl_data["machine_prompt"], str):
                    text = self.pc_prefix + pkl_data["machine_prompt"]
                if isinstance(pkl_data["machine_prompt"], list):
                    text = self.pc_prefix + pkl_data["machine_prompt"][0]
            else: 
                # blip caption
                caption_key = "/".join(pkl_data["item_path"].split("/")[-2:])
                text = self.pc_prefix + self.sun_blip_captions[caption_key][0]

            text = " ".join(text.split(" ")[:60])
            text = text[:300]
            
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

    ### Read caption ###
    def _get_objaverse_caption_(self, pkl_data):
        if self.load_interlm:
            text = self.pc_prefix + self.interlm_caption[pkl_data["name"]]
        elif "human_prompt" in pkl_data.keys():
            text = self.pc_prefix + pkl_data["human_prompt"][0]
        elif self.load_blip2:
            text = self.pc_prefix + self.blip2_caption[pkl_data["name"]][0]
        elif "machine_prompt" in pkl_data.keys():
            text = self.pc_prefix + pkl_data["machine_prompt"]
        else:
            text= ""
        
        if len(text)>315: text=""
        return text

    def _get_sunrgbd_caption(self, pkl_data):
        if self.load_blip2:
            # blip caption
            caption_key = "/".join(pkl_data["item_path"].split("/")[-2:])
            text = self.pc_prefix + self.sun_blip_captions[caption_key][0]
        else:
            # SUN RGB-D original caption
            text = self.pc_prefix + pkl_data["label"]
        return text
    
    def _getitem_3dgs_(self, pkl_data):
        #  51407 | 139282, R:[-1.92, -1.70, 0.25, 1.92, 6.35], G:[-1.92, -1.71, 0.14, 1.78, 6.50], B:[-1.92, -1.73, 0.02, 1.73, 6.31]

        if self.load_npy:
            item_id = pkl_data["name"]
            npy_path = self.item_id_to_pts[item_id]
            xyz = np.load(npy_path, allow_pickle=True)
            x,y,z = xyz[:,0:1], xyz[:,1:2], xyz[:,2:3]
            # xyz = np.concatenate((x,z,-y), -1)
            xyz = np.concatenate((x,y,z), -1)

            pkl_gaussians = np.concatenate((xyz, np.ones((xyz.shape[0], 11))*0.4), -1)
            pkl_gaussians = torch.from_numpy(pkl_gaussians).to(torch.float32)

            ran_sel = np.random.randint(0, pkl_gaussians.shape[0], 1_024)
            pkl_gaussians = pkl_gaussians[ran_sel]
        
        else:
            pkl_gaussians = torch.from_numpy(pkl_data["3dgs"])
            pkl_gaussians = self.expand_3dgs(pkl_gaussians)

            # y,z = pkl_gaussians[:,1:2].clone(), pkl_gaussians[:,2:3].clone()
            # pkl_gaussians[:,1:2] = z
            # pkl_gaussians[:,2:3] = -y

            pkl_gaussians[:,3:] = torch.tanh(pkl_gaussians[:,3:])
            # pkl_gaussians[:,6:] = 0


            if self.using_channel>0:
                pkl_gaussians = pkl_gaussians[:,:self.using_channel]

        return pkl_gaussians.transpose(0,1)

    def _get_abo_caption_(self, pkl_data):
        text = self.pc_prefix + pkl_data["label"]
        text = text[:300]
        return text

    def _getitem_extra_(self, pkl_data, meta_data):
        if "label_count" in pkl_data.keys():
            meta_data["label_count"] = pkl_data["label_count"]
            if self.update_label:
                meta_data["label_count"] = self.update_label_json[pkl_data["label"]]
        else: meta_data["label_count"] = 0
        meta_data["valid"] = (meta_data['text'] != "")
        return meta_data

def get_Objaverse_dataloader(
        data_dir,
        file_paths_json,
        batch_size=1,
        nw=0,
        shuffle=True,
        is_train=True,
        prefetch_factor = 2,
        distributed=False,
        **kwargs,
        ):
    
    sampler = None
    Objaverse_dataset = Objaverse(data_dir,file_paths_json,shuffle=shuffle,is_train=is_train,**kwargs)
    if distributed:
        sampler = DistributedSampler(Objaverse_dataset)
        
    return DataLoader(Objaverse_dataset,batch_size,shuffle=shuffle,pin_memory=True,drop_last=False,num_workers=nw,
                      prefetch_factor=prefetch_factor,sampler=sampler)

def show_dict(data:dict,prefix=""):
    for key in data.keys():
        if hasattr(data[key],"shape"):
            print(prefix,key,data[key].shape)
        elif type(data[key]) == dict:
            print(key)
            show_dict(data[key],"\t")
        elif type(data[key]) in [bool,float,int]:
            print(key)
        elif type(data[key]) in [str]:
            print(key,data[key])
        else :
            print(prefix,key,len(data[key]))


if __name__ == "__main__":
    data_dir = "/path/to/your/gaussian-splatting/unigs/objaverse_all"

    dataset = Objaverse(data_dir, is_train=False)
    dataset.dump_split_file("sample_all.json", 
                            sample_num=100000, 
                            keep_original=True,
                            keep_val_split=False,
                            )

    # dataset = get_Objaverse_dataloader(data_dir,batch_size=2)
    # dataset = get_Objaverse_dataloader(data_dir,is_train=False)

    # _xyz: [1024,3]
    # _features_dc: [1024,1,3]
    # _opacity: [1024,1]
    # _scaling: [1024,3]
    # _rotation: [1024,4]

    exit()
    count = 1
    for data in dataset:
        # print(count)
        count += 1

        print(data["gaussian"].shape)

        exit()
        pass
