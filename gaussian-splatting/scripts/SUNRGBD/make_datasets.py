import os
from sunrgbd import SUN
from tqdm import tqdm
import json
import torch

# nohup python /path/to/your/gaussian-splatting/scripts/make_datasets.py > "/path/to/your/gaussian-splatting/scripts/running.log" 2>&1 &

# dataset item
meta_file_path = "//path/to/your/SUNRGBD/SUNRGBDMeta3DBB_v2.mat"
meta_file_2D_path = "//path/to/your/SUNRGBD/SUNRGBDMeta2DBB_v2.mat"
root = "//path/to/your/SUNRGBD"
sun_data = SUN(meta_file_path, meta_file_2D_path, root, using_fbx=True)

# data id
kv1Index=sun_data.getSensorDataId('kv1')
kv2Index=sun_data.getSensorDataId('kv2')
realsenseIndex=sun_data.getSensorDataId('realsense')
xtionIndex=sun_data.getSensorDataId('xtion')
allIndex = [*kv1Index, *kv2Index, *realsenseIndex, *xtionIndex] # 10335

# data type
using_37seglist = False
data_type_json_path = "//path/to/your/SUNRGBD/label.json"
with open(data_type_json_path, "r") as file:
    data_type_json = json.load(file)

skip_zero_folder = True
replacing_dic = {
    "addingmachine":"adding machine",
    "air_condition":"air conditioner",
    "airconditioner":"air conditioner",
    "air_conditioner":"air conditioner",
    "airfan":"air fan",
    "airpot":"air pot",
    "attachecase":"attache case",
    "basketballhoop":"basketball hoop",
    "bathtowel":"bath towel",
    "bedding":"bed",
    "bulletinboard":"bulletin board",
    "bulleting_board":"bulletin board",
    "canopener":"can opener",
    "cautionsign":"caution sign",
    "cdcassette":"cd cassette",
    "cellphone":"cell phone",
    "chalk_board":"chalkboard",
    "chairs":"chair",
    "chargingstation":"charging station",
    "choppingboard":"chopping board",
    "circuitbreaker":"circuit breaker",
    "coffeetable":"coffee table",
    "computer_monitor_and_keyboard":"computer monitor with keyboard",
    "computerrack":"computer rack",
    "dish_washing_liquid":"dishwashing liquid",
    "displaycase":"display case",
    "doormat":"door mat",
    "drying_rack":"dryingrack",
    "dvdplayer":"dvd player",
    "emergencylight":"emergency light",
    "endtable":"end table",
    "entable":"end table",
    "fire_extinguiser":"fire extinguisher",
    "flowerbase":"flower base",
    "fryingpan":"frying pan",
    "fumehood":"fume hood",
    "grandfatherclock":"grandfather clock",
    "grandpiano":"grand piano",
    "hair_brush":"hairbrush",
    "hand_bag":"handbag",
    "holepuncher":"hole puncher",
    "key_board":"keyboard",
    "laundrybasket":"laundry basket",
    "light_fixture":"lighting fixture",
    "messengerbag":"messenger bag",
    "mouse_&_mouse_pad":"mouse and mouse pad",
    "mouse_mouse_pad":"mouse and mouse pad",
    "mousepad":"mouse pad",
    "night_stand":"nightstand",
    "paperbag":"paper bag",
    "papercutter":"paper cutter",
    "paperholder":"paper holder",
    "papertowel":"paper towel",
    "papertoweldispenser":"paper towel dispenser",
    "papertowels":"paper towels",
    "papertray":"paper tray",
    "paperwrap":"paper wrap",
    "pictureframe":"picture frame",
    "pingpongtable":"ping pong table",
    "remote_conrol":"remote control",
    "remotecontrol":"remote control",
    "ricecooker":"rice cooker",
    "rubbermat":"rubber mat",
    "scissor":"scissors",
    "showercurtain":"shower curtain",
    "showerroom":"shower room",
    "sidetable":"side table",
    "sigh":"sign",
    "smokeabsorber":"smoke absorber",
    "soapdispenser":"soap dispenser",
    "soaptray":"soap tray",
    "standboard":"stand board",
    "stepstool":"step stool",
    "toasteroven":"toaster oven",
    "toyhouse":"toy house",
    "venetianblinds":"venetian blinds",
    "waterdispenser":"water dispenser",
    "waterheater":"water heater",
}
excluding_list=["cdoor","coun","unknown","waterjag"] # meaningless words
skipping_list = [
    "shelr","brawer","flower","file box","manual","laminator machine","lamination machine","urinal bowl",
    "toilet seat cover dispenser","floor sign","soccer table","dish detergent","game console","booklet","cook top",
    "laptop case","decoration","decorations","cream","sweep","salt","cooking pot","apron","wok","laundry rack","neck tie",
    "wreath","pot holder","chocolate box","laptop cover","water filter","scarf","oven mitt","water bottle pack","shoe box",
    "skillet","washer","kerosene","hose","ceiling fan","toilet paper holder","grapes","oranges","fuse box","carved display",
    "sheet","floral display","mobile phone","wristwatch","quilt","soap dish","shower hose","shower nozzle","radiator","dopp kit",
    "soap holder","dish","hairdryer","toiletries","toilet brush","toothbrush kit","comb","deodorant","pouch bag","soap and soap dish",
    "cup with toothbrush","gym bag","mini shelf","luggage stool","safe","razor","hangers","assorted boxes","box with items",
    "laboratory table","sealer","machine stand","boxes","gloves","dooe",
] # words with no 3D bbo

folder_names_10 = [
    "air conditioner","alarm","armoire","back pack","bag","bags","basin","basket","bathtub","battery","bed","bed sheet",
    "bench","bicycle","binder","blackboard","blanket","blender","blinds","board","book","books","bookshelf","bookstand",
    "bottle","bowl","box","brick","bucket","bulletin","bulletin board","bunk bed","cabinet","can","carpet","cart","cartoon",
    "chair","chalkboard","chest","child chair","clock","closet","cloth","clothing rack","coffee maker","coffee table",
    "coffeemaker","computer","computer keyboard","container","counter","cpu","crib","cubby","cup","cupboard","cups","curtain",
    "decor","desk","desktop","dining table","diploma","dishwasher","doll","door","drawer","dresser","dresser mirror",
    "dryingrack","easel","electric fan","end table","eraser","fan","faucet","figurine","fire extinguisher","fire place",
    "flower vase","flowers","fridge","frige","fume hood","futon","garbage bin","glass","grab bar","hamper","hand dryer",
    "headphones","headset","heater","helmet","information board","island","jacket","jar","jug","kakejiku","kettle","keyboard",
    "kitchen","ladder","lamp","laptop","laundry basket","light","lighting fixture","locker","machine","magazine","magazine rack"
    ,"map","mattress","microwave","microwave oven","mirror","monitor","mouse","mouse pad","mug","newspaper","nightstand","notebook"
    ,"organizer","ottoman","outlet","oven","packet","painting","pan","paper","paper bag","paper towel","paper towel dispenser",
    "parcel","pen","person","phone","piano","piano bench","picture","pillow","pipe","pitcher","pizza box","plant","plants",
    "plastic bag","plate","player","podium","portrait","poster","pot","printer","projector","projector screen","rack","recycle bin",
    "remote","remote control","rice cooker","rug","sandal","saucer chair","scanner","shampoo","shelf","shoe","shoe rack","shoes",
    "shower curtain","side table","sign","sink","soap dispenser","sofa","sofa bed","sofa chair","speaker","stack of chairs","stand",
    "stapler","step stool","stool","stove","stuffed toy","styrofoam","suits case","switch","table","telephone","thermos","tissue",
    "tissuebox","toaster","toaster oven","toilet","toilet paper","toilet paper dispenser","towel","towel holder","toy","toy car",
    "toys","tray","tripod","tupperware","tv","tv stand","urinal","vacuum cleaner","vanity","vase","washing machine","water bottle",
    "water dispenser","water fountain","water jug","whiteboard","window","window shade","wire container",
]
folder_names_5 = [
    "air conditioner",
    "alarm",
    "armoire",
    "baby chair",
    "back pack",
    "bag",
    "bags",
    "banana",
    "banner",
    "basin",
    "basket",
    "bathmat",
    "bathtub",
    "battery",
    "bed",
    "bed sheet",
    "bench",
    "bicycle",
    "bin",
    "binder",
    "blackboard",
    "blanket",
    "blender",
    "blinds",
    "board",
    "book",
    "books",
    "bookshelf",
    "bookstand",
    "bottle",
    "bottles",
    "bowl",
    "box",
    "brick",
    "brush",
    "bucket",
    "bulletin",
    "bulletin board",
    "bunk bed",
    "cabinet",
    "calendar",
    "camera",
    "can",
    "carpet",
    "cart",
    "cartoon",
    "case",
    "cell phone",
    "chair",
    "chalkboard",
    "chandelier",
    "charger",
    "chest",
    "child chair",
    "chopping board",
    "clock",
    "closet",
    "cloth",
    "clothing rack",
    "coat",
    "coat rack",
    "coffee cup",
    "coffee maker",
    "coffee table",
    "coffeemaker",
    "computer",
    "computer keyboard",
    "computer mouse",
    "condiments",
    "container",
    "counter",
    "cpu",
    "cpu case",
    "crib",
    "cubby",
    "cup",
    "cupboard",
    "cups",
    "curtain",
    "decor",
    "desk",
    "desktop",
    "dining table",
    "diploma",
    "dish soap",
    "dishdetergent",
    "dishwasher",
    "dishwashing liquid",
    "dispenser",
    "display",
    "display rack",
    "divider",
    "document holder",
    "doll",
    "door",
    "door knob",
    "drawer",
    "dresser",
    "dresser mirror",
    "drum",
    "dryer",
    "dryingrack",
    "dvd player",
    "easel",
    "electric fan",
    "electric pot",
    "end table",
    "eraser",
    "espresso machine",
    "extension wire",
    "fan",
    "faucet",
    "fax",
    "figurine",
    "fire extinguisher",
    "fire place",
    "flower pot",
    "flower vase",
    "flowers",
    "folder",
    "food",
    "food tray",
    "foosball table",
    "fridge",
    "frige",
    "frying pan",
    "fume hood",
    "futon",
    "garbage bin",
    "glass",
    "globe",
    "grab bar",
    "grill",
    "guitar",
    "hamper",
    "hand dryer",
    "hanger",
    "headboard",
    "headphones",
    "headset",
    "heater",
    "helmet",
    "high chair",
    "information board",
    "island",
    "jacket",
    "jar",
    "jug",
    "kakejiku",
    "kettle",
    "keyboard",
    "kitchen",
    "knives",
    "lab centrifuge",
    "ladder",
    "lamp",
    "laptop",
    "laundry basket",
    "lego",
    "light",
    "lighting fixture",
    "locker",
    "lotion",
    "lounge chair",
    "lumber",
    "machine",
    "magazine",
    "magazine rack",
    "map",
    "marker",
    "mat",
    "mattress",
    "meter",
    "microscope",
    "microwave",
    "microwave oven",
    "mini refrigerator",
    "mirror",
    "monitor",
    "mouse",
    "mouse pad",
    "mug",
    "newspaper",
    "nightstand",
    "notebook",
    "organizer",
    "ottoman",
    "outlet",
    "oven",
    "oven toaster",
    "package",
    "packet",
    "pad",
    "pail",
    "painting",
    "pan",
    "paper",
    "paper bag",
    "paper ream",
    "paper towel",
    "paper towel dispenser",
    "parcel",
    "pen",
    "person",
    "phone",
    "piano",
    "piano bench",
    "picture",
    "pillow",
    "ping pong table",
    "pipe",
    "pitcher",
    "pizza box",
    "placemat",
    "plant",
    "plants",
    "plastic",
    "plastic bag",
    "plate",
    "player",
    "plywood",
    "podium",
    "portrait",
    "poster",
    "pot",
    "power strip",
    "printer",
    "projector",
    "projector screen",
    "puncher",
    "rack",
    "radio",
    "recliner",
    "recycle bin",
    "remote",
    "remote control",
    "rice cooker",
    "roller",
    "rug",
    "sandal",
    "saucer chair",
    "scanner",
    "scissors",
    "shampoo",
    "shelf",
    "shoe",
    "shoe rack",
    "shoes",
    "shower",
    "shower curtain",
    "side table",
    "sign",
    "sink",
    "soap dispenser",
    "sofa",
    "sofa bed",
    "sofa chair",
    "speaker",
    "spoonholder",
    "spray",
    "stack of chairs",
    "stand",
    "stapler",
    "statue",
    "step stool",
    "stool",
    "stove",
    "stroller",
    "stuffed toy",
    "styrofoam",
    "suits case",
    "switch",
    "table",
    "tank",
    "tap",
    "tape",
    "telephone",
    "thermos",
    "tissue",
    "tissuebox",
    "toaster",
    "toaster oven",
    "toilet",
    "toilet paper",
    "toilet paper dispenser",
    "toiletpaper",
    "tool",
    "toolbox",
    "towel",
    "towel holder",
    "toy",
    "toy car",
    "toy house",
    "toys",
    "tray",
    "tree",
    "tripod",
    "trophy",
    "tube",
    "tupperware",
    "tv",
    "tv stand",
    "umbrella",
    "urinal",
    "vacuum cleaner",
    "vanity",
    "vase",
    "vending machine",
    "wall painting",
    "walldecor",
    "washing machine",
    "water bottle",
    "water dispenser",
    "water fountain",
    "water heater",
    "water jug",
    "whiteboard",
    "window",
    "window shade",
    "wire container",
    "workout equipment",
]


item_name_reflecting_dict = {}

print("reading data type ...")
data_label = {}
file_paths = {
    "original":{
        "train":[], 
        "test":[],
        "val":[]
    },
    "all":{},
}
if using_37seglist:
    for item_value in data_type_json.values():
        if item_value == "others": continue
        data_label[item_value] = 0
        file_paths["all"][item_value] = []

else:
    for item_id in tqdm(allIndex):
        cornerList2D, classNameList2D =sun_data.getCornerList2D(item_id)
        for class_name in classNameList2D:
            if "/" in class_name : continue

            if class_name in replacing_dic.keys():
                item_name = replacing_dic[class_name]
            else:
                item_name = str(class_name).replace(" ","").replace("_"," ").replace("-"," ")

            if item_name in excluding_list: continue
            
            if class_name not in item_name_reflecting_dict.keys():
                item_name_reflecting_dict[class_name] = item_name
            
            if skip_zero_folder and item_name in skipping_list: continue

            if item_name not in data_label.keys():
                data_label[item_name] = 0
                file_paths["all"][item_name] = []

# data split
raw_data_root = "/path/to/your/gaussian-splatting/clip3/raw_scene"
data_split_json_path = os.path.join(raw_data_root, "split.json")
with open(data_split_json_path, "r") as file:
    data_split = json.load(file)

# make dirs
save_data_root = "/path/to/your/gaussian-splatting/clip3/sunrgbd_all" ##########
os.makedirs(save_data_root, exist_ok=True)
for data_type in data_label.keys():
    data_type_root = os.path.join(save_data_root, data_type)
    os.makedirs(data_type_root, exist_ok=True)


# load data and split
raw_data_names = os.listdir(raw_data_root)
if "split.json" in raw_data_names: raw_data_names.remove("split.json")
raw_data_names.sort(key=lambda x: int(x.split(".")[0].split("_")[1]) )

print("splitting data ...")
for raw_data_name in raw_data_names:
    if not raw_data_name.endswith(".pkl"): continue
    print("Loading raw data %s ..."%(raw_data_name))
    raw_data = torch.load(os.path.join(raw_data_root, raw_data_name))

    for scene_data in tqdm(raw_data, desc=raw_data_name):
        sequence_name = scene_data['name']
        data_split = scene_data['split']
        for i,item_name in enumerate(scene_data['className2D']):
            if using_37seglist:
                label_type_count = scene_data["label_count"][i]
                if label_type_count == 0: continue

                item_name = data_type_json[str(label_type_count)]
            else:
                if "/" in item_name or item_name in excluding_list: continue
                item_name = item_name_reflecting_dict[item_name]
                if skip_zero_folder and item_name in skipping_list: continue
                # if not item_name in folder_names_10: continue

            if scene_data["img"][i].shape[0]<10 or scene_data["img"][i].shape[1]<10: continue
            if scene_data["3dgs"][i].shape[0]<100 : continue

            pkl_name = str(data_label[item_name])+".pkl"
            save_item_path = os.path.join(save_data_root, "objects", item_name, pkl_name)
            file_paths["original"][data_split].append(save_item_path)
            file_paths["all"][item_name].append(save_item_path)
            data_label[item_name] += 1

            # if os.path.exists(save_item_path): continue

            save_item_data = {
                "name":sequence_name,
                "item_path":save_item_path,
                "label_count":scene_data["label_count"][i],
                "dataset": "sunrgbgd",
                "label":data_type_json[str(scene_data["label_count"][i])],
                "img":scene_data["img"][i],
                "3dgs":scene_data["3dgs"][i],
                
                "className2D":scene_data["className2D"][i],
                "className3D":scene_data["className3D"][i][0]
            }
            torch.save(save_item_data, save_item_path)
    
    del raw_data
    # break
    # exit()

with open(os.path.join(save_data_root, "file_paths.json"), "w") as file:
    json.dump(file_paths, file, indent=4)
