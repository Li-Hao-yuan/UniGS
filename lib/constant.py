import json
from eval import test_retrive

class Constant:

    # sunrgbd
    SUNRGBD_type_list = ["wall","floor","cabinet","bed","chair","sofa","table","door",
                    "window","bookshelf","picture","counter","blinds","desks","shelves",
                    "curtain","dresser","pillow","mirror","floor-mat","clothes","ceiling",
                    "books","refrigerator","television","paper","towel","shower-curtain",
                    "box","whiteboard","person","nightstand","toilet","sink","lamp","bathtub","bag"]
    SUNRGBD_collect_list = ["bed","bookshelf","chair","desks","sofa","table","toilet","bathtub","dresser","nightstand"]

    # ABO
    ABO_type_list = ["bed","bench","cabinet","chair","desk","dresser","furniture","home","lamp","mirror","ottoman",
                    "pillow","planter","rug","shelf","sofa","stool","storage","table","vase","wall_art"]
    # ABO_collect_list = ["bed","bookshelf","chair","desks","sofa","table","toilet","bathtub","dresser","nightstand"]
    ABO_collect_list = ["bed","bench","cabinet","chair","desk","dresser","furniture","home","lamp","mirror","ottoman",
                    "pillow","planter","rug","shelf","sofa","stool","storage","table","vase","wall_art"]
    
    # Objaverse
    Objaverse_type_list = []
    lvis_annotation_path = "/path/to/your/objarverse/openshape/meta_data/split/lvis.json"
    with open(lvis_annotation_path, "r") as file:
        lvis_content = json.load(file)
    for lvis_item in lvis_content:
        if lvis_item["category"] not in Objaverse_type_list: Objaverse_type_list.append(lvis_item["category"])
    Objaverse_collect_list = ["mug","owl","mushroom","fireplug","banana","ring","doughnut","armor","sword","control","cone",
                        "gravestone","chandelier","snowman","shield","antenna","seashell","chair",]
    
    # mvimgnet
    Mvimgnet_annotation_path = "/path/to/your/MVimgnet/scripts/mvimgnet_category.txt"
    Mvimgnet_type_list = []
    with open(Mvimgnet_annotation_path, "r") as file:
        for line in file.readlines():
            line = line.replace("\n","").split(",")
            Mvimgnet_type_list.append(line[1].lower())
    Mvimgnet_collect_list = ['bottle', 'conch', 'tangerine', 'okra', 'guava', 'bulb', 'bag', 'glove', 
                              'accessory', 'garlic', 'lipstick', 'telephone', 'watch', 'lock', 'bowl', 'toothpaste']

    classification_mapping = {
        "sunrgbd":[SUNRGBD_type_list, SUNRGBD_collect_list],
        "abo":[ABO_type_list, ABO_collect_list],
        "objaverse":[Objaverse_type_list, Objaverse_collect_list],
        "mvimgnet":[Mvimgnet_type_list, Mvimgnet_collect_list]
    }
    retrive_mapping = {
        "sunrgbd":[[],[]],
        "abo":[[],[]],
        "objaverse":[[],[]],
        'mvimgnet':[[],[]]
    }

    task_mapping = {
        "retrive": retrive_mapping,
        'classification': classification_mapping,
    }

    eval_func_mapping = {
        "retrive": test_retrive,
        'classification': None,
    }
    
    def __init__(self,
                 task, 
                 dataset,
                 pc_prefix = "point cloud of "
                 ) -> None:
        assert task in ["retrive", "classification"]
        assert dataset in ["sunrgbd", "abo", "objaverse", "mvimgnet"]

        self.task = task
        self.dataset = dataset
        self.pc_prefix = pc_prefix

        self.data_type_list, self.collect_data_list = self.task_mapping[task][dataset]
        self.test_func = self.eval_func_mapping[task]
    
    def get_data_type_list(self):
        return self.data_type_list

    def get_collect_data_list(self):
        return self.collect_data_list

    def get_test_func(self):
        return self.test_func

    def get_testing_text(self):
        if self.task == "retrive":
            return None
        elif self.task == "classification":
            testing_text = []
            for i in range(len(self.data_type_list)):
                testing_text.append(self.pc_prefix+self.data_type_list[i])
            return testing_text
        else:
            raise RuntimeError("No such task!")

