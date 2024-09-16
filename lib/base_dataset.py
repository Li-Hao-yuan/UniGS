import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision
    
class BaseDataset(Dataset):

    def __init__(self,
        data_dir:str,
        data_split:str="file_paths.json",
        resolution:int=224,
        shuffle:bool=True,
        pc_prefix:str="point cloud of ",
        is_train:bool=True,
        using_channel:int=-1,
        load_extend=False,
        load_npy=False,
        ):

        super().__init__()
        self.data_dir = data_dir
        self.is_train = is_train

        self.pc_prefix = pc_prefix
        self.using_channel = using_channel
        
        content = self.read_file_paths(data_dir, data_split)

        try:
            if is_train: 
                self.file_paths = content["train"]
                if shuffle: random.shuffle(self.file_paths)
            else: 
                self.file_paths = content["test"]
        except:
            self.file_paths = []
        
        self.file_paths_copy = self.file_paths.copy()

        self.test_text = None
        self.load_npy = load_npy
        print("load_npy:", load_npy)
        self.transform =  torchvision.transforms.Compose([
            torchvision.transforms.Resize((resolution,resolution), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) 
        self.load_extend = load_extend
    
    def dump_split_file(self, dump_file_paths, specific_keys=[], sample_num=10000, 
                        ratio=0.8, overlap=False, keep_original=True, keep_val_split=False):
        file_paths_json = os.path.join(self.data_dir, "file_paths.json")
        if keep_original:
            with open(file_paths_json, "r") as file:
                content = json.load(file)
                content["train"] = content["original"]["train"]
                if not keep_val_split: 
                    content["train"].extend(content["original"]["val"])
                content["test"] = content["original"]["test"]
                content.pop("original")
            file_paths_json = os.path.join(self.data_dir, dump_file_paths)
            with open(file_paths_json, "w") as file:
                json.dump(content, file, indent=4)
            exit()
        else:

            with open(file_paths_json, "r") as file:
                content = json.load(file)
                content["train"] = []
                content["test"] = []
                content.pop("original")
            
            for key in content["all"].keys():
                item_type_paths = content["all"][key]
                random.shuffle(item_type_paths)
                item_type_paths = item_type_paths[:sample_num]

                if key not in specific_keys:
                    content["train"].extend(item_type_paths)
                    continue

                train_test_ptr = int(len(item_type_paths)*ratio)

                if overlap:
                    content["train"].extend(item_type_paths)
                    content["test"].extend(item_type_paths[train_test_ptr:])
                else:
                    content["train"].extend(item_type_paths[:train_test_ptr])
                    content["test"].extend(item_type_paths[train_test_ptr:])
            
            content.pop("all")
            file_paths_json = os.path.join(self.data_dir, dump_file_paths)
            with open(file_paths_json, "w") as file:
                json.dump(content, file, indent=4)
            
            exit()

    def read_file_paths(self, data_dir, data_split="file_paths.json"):
        file_paths_json = os.path.join(data_dir, data_split)
        with open(file_paths_json, "r") as file:
            content = json.load(file)
        return content

    def expand_3dgs(self, guassians, clip_num=1024):
        '''
        3dgs : [n, 14]
        '''
        pts_num = guassians.shape[0]
        if pts_num == clip_num:
            return guassians
        elif pts_num<clip_num: 
            guassians = guassians.repeat( int(clip_num/pts_num)+1 ,1)
            return guassians[:clip_num]
        else:
            data_index = np.arange(pts_num)
            random.shuffle(data_index)
            guassians = guassians[data_index[:clip_num]]
            return guassians

    ### Read image ###
    def _get_objaverse_image_(self, pkl_data):
        exist_img_names = os.listdir(pkl_data["img"])
        img_names = []
        for img_name in exist_img_names:
            if "npy" not in img_name and os.path.isfile(os.path.join(pkl_data["img"],img_name)): 
                img_names.append(img_name)
        img_path = os.path.join(pkl_data["img"], img_names[np.random.randint(len(img_names))])
        image = Image.open(img_path).convert("RGB")
        return image

    def _get_sunrgbd_image_(self, pkl_data):
        image = Image.fromarray(pkl_data["img"])
        return image

    def _get_abo_image_(self, pkl_data):
        exist_img_names = os.listdir(pkl_data["img"])
        img_names = []
        for img_name in exist_img_names:
            if "json" not in img_name: img_names.append(img_name)
        img_path = os.path.join(pkl_data["img"], img_names[np.random.randint(len(img_names))])
        image = Image.open(img_path).convert("RGB")
        return image

    def _get_mvimgnet_image_(self, pkl_data):
        exist_img_names = os.listdir(pkl_data["img"])
        img_names = []
        for img_name in exist_img_names:
            if "bg_removed" not in img_name: img_names.append(img_name)
        img_path = os.path.join(pkl_data["img"], img_names[np.random.randint(len(img_names))])
        image = Image.open(img_path).convert("RGB")
        return image

    def _getitem_image_(self, pkl_data):
        if "dataset" in pkl_data.keys(): 
            dataset_caption_func = {
                "sunrgbd": self._get_sunrgbd_image_,
                "abo": self._get_abo_image_,
                "objaverse": self._get_objaverse_image_,
                "mvimgnet": self._get_mvimgnet_image_,
            }
            image = dataset_caption_func[pkl_data['dataset']](pkl_data)
        else:
            image = Image.fromarray(pkl_data["img"])
        return self.transform(image)
        
    ### Read caption ###
    def _get_objaverse_caption_(self, pkl_data):
        if "human_prompt" in pkl_data.keys():
            text = self.pc_prefix + pkl_data["human_prompt"][0]
        elif "machine_prompt" in pkl_data.keys():
            text = self.pc_prefix + pkl_data["machine_prompt"]
        return text

    def _get_sunrgbd_caption_(self, pkl_data):
        return self.pc_prefix + pkl_data["label"]

    def _get_abo_caption_(self, pkl_data):
        text = self.pc_prefix + pkl_data["machine_prompt"][0]
        text = text[:310]
        text = " ".join(text.split(" ")[:50])
        return text

    def _get_mvimgnet_caption_(self, pkl_data):
        return self.pc_prefix + pkl_data["label"]

    def _getitem_caption_(self, pkl_data):
        if "dataset" in pkl_data.keys(): 
            dataset_caption_func = {
                "sunrgbd": self._get_sunrgbd_caption_,
                "abo": self._get_abo_caption_,
                "objaverse": self._get_objaverse_caption_,
                "mvimgnet": self._get_mvimgnet_caption_,
            }
            text = dataset_caption_func[pkl_data['dataset']](pkl_data)
        else:
            text = self.pc_prefix + pkl_data["label"]
        text = text[:320]
        return text
    
    def _getitem_3dgs_(self, pkl_data):
        #  51407 | 139282, R:[-1.92, -1.70, 0.25, 1.92, 6.35], G:[-1.92, -1.71, 0.14, 1.78, 6.50], B:[-1.92, -1.73, 0.02, 1.73, 6.31]

        if self.load_npy:
            item_id = pkl_data["name"]
            npy_path = self.item_id_to_pts[item_id]
            xyz = np.load(npy_path, allow_pickle=True)
            pkl_gaussians = np.concatenate((xyz, np.ones((xyz.shape[0], 11))*0.4), -1)
            pkl_gaussians = torch.from_numpy(pkl_gaussians).to(torch.float32)

            ran_sel = np.random.randint(0, pkl_gaussians.shape[0], 1_024)
            pkl_gaussians = pkl_gaussians[ran_sel]
        
        else:
            pkl_gaussians = torch.from_numpy(pkl_data["3dgs"])
            pkl_gaussians = self.expand_3dgs(pkl_gaussians)

            pkl_gaussians[:,3:] = torch.tanh(pkl_gaussians[:,3:])
            # pkl_gaussians[:,6:] = 0


            if self.using_channel>0:
                pkl_gaussians = pkl_gaussians[:,:self.using_channel]

        return pkl_gaussians.transpose(0,1)

    ### extra ###
    def _getitem_extra_(self, pkl_data, meta_data):
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

    def __len__(self):
        return len(self.file_paths)


