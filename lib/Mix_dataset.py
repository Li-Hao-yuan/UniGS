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

class MixDataset(BaseDataset):
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
        load_interlm=True,
        **kwargs
        ):

        super().__init__(data_dir, data_split, resolution, shuffle, pc_prefix, is_train, using_channel, False)
            
        self.load_interlm = load_interlm
        if load_interlm:
            print("Dataset load InterLM as caption!")
            interlm_caption_path = "/path/to/your/objarverse/InternLM_xcomposer2_caption.json"
            with open(interlm_caption_path, "r") as file:
                self.objaverse_interlm_caption = json.load(file)
            
            interlm_caption_path = "/path/to/your/ABO/InternLM_xcomposer2_caption.json"
            with open(interlm_caption_path, "r") as file:
                self.abo_interlm_caption = json.load(file)

            interlm_caption_path = "/path/to/your/MVimgnet/InternLM_xcomposer2_caption.json"
            with open(interlm_caption_path, "r") as file:
                self.mvimgnet_interlm_caption = json.load(file)
            
    
    def gather_retrive_prompts(self):
        text_list, item_paths, text_item_dict = [], [], {}
        for index in range(len(self.file_paths)):
            pkl_path = self.file_paths[index]
            pkl_data = torch.load(pkl_path)

            if pkl_data["dataset"] == "objaverse":
                text = self._get_objaverse_caption_(pkl_data)
            elif pkl_data["dataset"] == "abo":
                text = self._get_abo_caption_(pkl_data)
            else:
                text = self._get_mvimgnet_caption_(pkl_data)
            
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

    def _get_mvimgnet_image_(self, pkl_data):
        img_root = pkl_data["img"].replace("/mvi/","/mvi_low/")
        exist_img_names = os.listdir(img_root)
        img_names = []
        for img_name in exist_img_names:
            if "bg_removed" not in img_name: img_names.append(img_name)
        img_path = os.path.join(img_root, img_names[np.random.randint(len(img_names))])
        image = Image.open(img_path).convert("RGB")
        return image

    ### Read caption ###
    def _get_objaverse_caption_(self, pkl_data):
        if self.load_interlm:
            text = self.pc_prefix + self.objaverse_interlm_caption[pkl_data["name"]]
        elif "human_prompt" in pkl_data.keys():
            text = self.pc_prefix + pkl_data["human_prompt"][0]
        elif "machine_prompt" in pkl_data.keys():
            text = self.pc_prefix + pkl_data["machine_prompt"]
        else:
            text= ""
        
        text = text[:315]
        return text

    def _get_abo_caption_(self, pkl_data):
        return self.pc_prefix + self.abo_interlm_caption[pkl_data["name"]]
    
    def _get_mvimgnet_caption_(self, pkl_data):
        return self.pc_prefix + self.mvimgnet_interlm_caption[pkl_data["name"]]

    def _getitem_extra_(self, pkl_data, meta_data):
        if "label_count" in pkl_data.keys():
            meta_data["label_count"] = int(pkl_data["label_count"])
        else: meta_data["label_count"] = 0
        # meta_data["valid"] = (meta_data['text'] != "")
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

if __name__ == "__main__":
    dataset = MixDataset("/path/to/your/gaussian-splatting/unigs/objaverse_all", "sample_all_datasets.json")
    dataset = DataLoader(dataset, 20)
    for i,data in enumerate(dataset):
        print(i,data["item_path"])