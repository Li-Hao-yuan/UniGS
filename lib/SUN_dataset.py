import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from base_dataset import BaseDataset

all_to_map_using_data_type = [
    "cabinet","bed","bunk bed","baby bed","chair","child chair","saucer chair","stack of chairs","high chair","lounge chair",
    "baby chair","bench","sofa","sofa bed","sofa chair","side table","table","coffee table","dining table","end table","foosball table",
    "ping pong table","door","window","bookshelf","picture","counter","cupboard","closet","blinds","desk","shelves", "shelf","window shade",
    "curtain","dresser","pillow","dresser mirror","mirror","mat","bathmat","carpet","cloth","coat","jacket","mini refrigerator","tv","newspaper",
    "paper","paper ream","paper towel","paper towel dispenser","tissuebox","toilet paper","toiletpaper","towel","shower curtain","box","pizza box",
    "whiteboard","bulletin board","blackboard","chalkboard","bulletin","board","person","nightstand","toilet","sink","lamp","light","lighting fixture",
    "bathtub","paper bag","plastic bag","bag","bags",
]
    
class SUN(BaseDataset):
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
        data_type_json_path = "/path/to/your/SUNRGBD/label_inverse.json",
        refine_label=True,
        load_npy=False,
        **kwargs
        ):

        super().__init__(data_dir, data_split, resolution, shuffle, pc_prefix, is_train, using_channel, load_npy)
        
        self.item_type = ["wall","floor","cabinet","bed","chair","sofa","table","door",
                          "window","bookShelf","picture","counter","blinds","desks","shelves",
                          "curtain","dresser","pillow","mirror","floor-mat","clothes","ceiling",
                          "books","refrigerator","television","paper","towel","shower-curtain",
                          "box","whiteboard","person","nightStand","toilet","sink","lamp","bathtub","bag"]
        
        self.refine_label = refine_label
        self.data_mapping = {
            "wall":[],
            "floor":[],
            "cabinet":["cabinet"],
            "bed":["bed","bunk bed","baby bed"],
            "chair":["chair","child chair","saucer chair","stack of chairs","high chair","lounge chair","baby chair","bench"],
            "sofa":["sofa","sofa bed","sofa chair"],
            "table":["side table","table","coffee table","dining table","end table","foosball table","ping pong table"],
            "door":["door"],
            "window":["window"],
            "bookShelf":["bookshelf"],
            "picture":["picture",],
            "counter":["counter","cupboard","closet",],
            "blinds":["blinds",],
            "desks":["desk"],
            "shelves":["shelves", "shelf",],
            "curtain":["window shade","curtain",],
            "dresser":["dresser",],
            "pillow":["pillow"],
            "mirror":["dresser mirror","mirror"],
            "floor-mat":["mat","bathmat","carpet"],
            "clothes":["cloth","coat","jacket"],
            "ceiling":[],
            "books":["notebook","book","books","magazine"],
            "refrigerator":["mini refrigerator"],
            "television":["tv"],
            "paper":["newspaper","paper","paper ream","paper towel","paper towel dispenser","tissuebox","toilet paper","toiletpaper"],
            "towel":["towel"],
            "shower-curtain":["shower curtain"],
            "box":["box","pizza box"],
            "whiteboard":["whiteboard","bulletin board","blackboard","chalkboard","bulletin","board"],
            "person":["person"],
            "nightStand":["nightstand"],
            "toilet":["toilet"],
            "sink":["sink"],
            "lamp":["lamp","light","lighting fixture"],
            "bathtub":["bathtub"],
            "bag":["paper bag","plastic bag","bag","bags"]
        }
        self.transfer_dict = {}
        for key in self.data_mapping.keys():
            for value in self.data_mapping[key]:
                self.transfer_dict[value] = key

        with open(data_type_json_path, "r") as file:
            self.data_type_json = json.load(file)

        if is_train and len(self.file_paths) > 0:
            self.reshape_file_paths()
        
        # self.file_paths = self.file_paths[::50]
        
    
    def reshape_file_paths(self):
        file_paths = self.file_paths
        file_path_dict = {key:[] for key in self.item_type}
        file_path_count = {key:0 for key in self.item_type}

        for file_path in file_paths:
            data_type = file_path.split("/")[-2]
            
            if data_type not in self.item_type:
                if data_type in self.transfer_dict.keys():
                    data_type = self.transfer_dict[data_type]
                else:
                    file_path_dict[data_type] = [file_path]
                    file_path_count[data_type] = 1
            

            file_path_dict[data_type].append(file_path)
            file_path_count[data_type] = 0
        
        file_path_rate = []
        for key in self.item_type:
            file_path_rate.append( len(file_path_dict[key])/len(self.file_paths) )
        file_path_rate = np.power(file_path_rate, 0.5)
        file_path_rate = file_path_rate/np.sum(file_path_rate)

        self.file_path_dict = file_path_dict
        self.file_path_rate = file_path_rate
        self.file_path_count = file_path_count

    def resample_file_paths(self):
        type_choice = random.choices(range(len(self.item_type)), self.file_path_rate)[0]

        data_type = self.item_type[type_choice]
        data_file_paths = self.file_path_dict[data_type]
        last_path_count = self.file_path_count[data_type]
        self.file_path_count[data_type] += 1

        file_paths_len = len(data_file_paths)
        choose_file_path = data_file_paths[last_path_count%(file_paths_len)]

        return choose_file_path

    def _getitem_extra_(self, pkl_data, meta_data, pkl_path):
        meta_data["label_count"] = int(pkl_data["label_count"])
        meta_data["valid"] = (pkl_data['dataset'] == "sunrgbd")

        data_type = pkl_path.split("/")[-2]
        if self.refine_label and data_type in self.transfer_dict.keys():
            if data_type not in self.item_type: 
                data_type = self.transfer_dict[data_type]

            meta_data["text"] = self.pc_prefix + data_type
            meta_data["label_count"] = int(self.data_type_json[data_type])

        return meta_data

    def _getitem_3dgs_(self, pkl_data):
        #  51407 | 139282, R:[-1.92, -1.70, 0.25, 1.92, 6.35], G:[-1.92, -1.71, 0.14, 1.78, 6.50], B:[-1.92, -1.73, 0.02, 1.73, 6.31]

        pkl_gaussians = torch.from_numpy(pkl_data["3dgs"])
        pkl_gaussians = self.expand_3dgs(pkl_gaussians)

        # x, y,z = pkl_gaussians[:,0:1].clone(), pkl_gaussians[:,1:2].clone(), pkl_gaussians[:,2:3].clone()
        # pkl_gaussians[:,0:1] = x
        # pkl_gaussians[:,1:2] = z
        # pkl_gaussians[:,2:3] = -y

        pkl_gaussians[:,3:] = torch.tanh(pkl_gaussians[:,3:])
        pkl_gaussians[:,3:6] = 0.4


        if self.using_channel>0:
            pkl_gaussians = pkl_gaussians[:,:self.using_channel]

        return pkl_gaussians.transpose(0,1)

    def __getitem__(self, index):
        '''
        name:                           <numpy.str_>
        img:            [325, 115, 3]   <numpy.ndarray>
        className2D:                    <numpy.str_>
        3dgs:           [13277, 14]     <numpy.ndarray>
        className3D:                    <numpy.str_> (old [1, ] | new [])
        '''
        pkl_path = self.file_paths[index]
        # if self.is_train: pkl_path = self.resample_file_paths()

        pkl_data = torch.load(pkl_path)
        meta_data = {}
        meta_data["image"] = self._getitem_image_(pkl_data)
        meta_data["gaussian"] = self._getitem_3dgs_(pkl_data)

        meta_data["text"] = self._getitem_caption_(pkl_data)

        meta_data["item_path"] = pkl_data["item_path"]

        meta_data = self._getitem_extra_(pkl_data, meta_data, pkl_path)

        return meta_data

    def __len__(self):
        return len(self.file_paths)


if __name__ == "__main__":
    data_dir = "/path/to/your/gaussian-splatting/unigs/sunrgbd_all"
    # data_dir = "/path/to/your/gaussian-splatting/unigs/sunrgbd"
    
    dataset = SUN(data_dir, data_split="sample_all.json", refine_label=True)
    # dataset = SUN(data_dir, is_train=False)
    # dataset.dump_split_file("sample_all.json", 
    #                         sample_num=100000, 
    #                         specific_keys=all_to_map_using_data_type
    #                         )

    # dataset = get_SUN_dataloader(data_dir,batch_size=2)
    # dataset = get_SUN_dataloader(data_dir,is_train=False)

    # max: [99973, '/path/to/your/gaussian-splatting/unigs/sunrgbd/train/picture/167.pkl', 'point cloud of a picture']
    # min: [100, '/path/to/your/gaussian-splatting/unigs/sunrgbd/train/lamp/8.pkl', 'point cloud of a lamp']

    # _xyz: [1024,3]
    # _features_dc: [1024,1,3]
    # _opacity: [1024,1]
    # _scaling: [1024,3]
    # _rotation: [1024,4]


    count = 1
    for data in dataset:
        # print(count)
        count += 1

        print(data["gaussian"].shape)

        exit()
        pass
'''
cabinet:            cabinet, display cabinet
bed:                baby bed, bed, bunk bed, child bed, folding bed,
chair:              baby chair, lawn chair, lounge chair, massage chair, rocking chair, saucer chair, high chair, chair, stack of chairs
sofa:               sofa, sofa bed, sofa chair
table:              long office table, coffee table, bar table, foosball table, pool table, table
door:               lift door, hingedoor, door
window:             glass window, window
bookShelf:          bookshelf
picture:            picture, pictures, picture frame
counter:            counter
blinds:             window shade, blinds, venetian blinds
desks:              desk
shelves:            shelves
curtain:            curtain blinds, curtain, 
dresser:            dresser
pillow:             pillow, 
mirror:             mirror, dresser mirror
floor-mat:          mat, mattress, rubber mat, door mat
books:              book, bookcase, books
refrigerator:       mini refrigerator
television:         television
paper:              poster paper, toilet paper, paper towel, paper towels
towel:              bath towel, towel
shower-curtain:     shower curtain
box:                plant box, pizza box, light box, storage box, 
whiteboard:         whiteboard
person:             person
nightStand:         nightstand
toilet:             toilet
sink:               sink, kitchen sink
lamp:               lamp, street lamp, walllamp
bathtub:            bathtub
'''