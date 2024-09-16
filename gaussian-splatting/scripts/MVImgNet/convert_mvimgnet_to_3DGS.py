import os
from tqdm import tqdm
import json

# no additonal 30%
# with mask(10)  32%
# all mask >90%

data_dir = "//path/to/your/MVimgnet/mvi"
mask_root = "//path/to/your/MVimgnet/mask"
save_root = "/path/to/your/gaussian-splatting/data/mvimgnet"
class_annotation_path = "//path/to/your/MVimgnet/scripts/mvimgnet_category.txt"
os.makedirs(save_root, exist_ok=True)

num_to_class = {}
with open(class_annotation_path, "r") as file:
    for line in file.readlines():
        line = line.replace("\n","").split(",")
        num_to_class[line[0]] = line[1].lower()

optimized_item_list = []
optimized_root = "/path/to/your/gaussian-splatting/data/mvimgnet_95"
for split in os.listdir(optimized_root):
    optimized_item_list.extend(os.listdir(os.path.join(optimized_root, split)))

move_data = True
count = [0, 0, 0]

for class_num in tqdm(os.listdir(data_dir)):
    class_type = num_to_class[class_num]

    for item_id in os.listdir(os.path.join(data_dir, class_num)):
        if item_id in optimized_item_list: continue
        os.makedirs(os.path.join(save_root, class_type), exist_ok=True)

        count[2] += 1
        item_root = os.path.join(data_dir, class_num, item_id)
        item_mask_root = os.path.join(mask_root, class_num, item_id)
        item_image_root = os.path.join(item_root, "images")
        image_names = os.listdir(item_image_root)
        image_names.sort()

        image_to_mask_paths = {}
        save_item_path = os.path.join(save_root, class_type, item_id)
        save_annotation_path = os.path.join(save_item_path, "type.txt")
        save_masks_root = os.path.join(save_item_path, "masks")

        if os.path.exists(item_mask_root):
            count[0] += 1

            for image_name in image_names:
                if "bg_removed" in image_name: continue
                mask_img_name = image_name+".png"
                mask_img_path = os.path.join(item_mask_root, mask_img_name)
                assert os.path.exists(mask_img_path), print(mask_img_path)
                image_to_mask_paths[os.path.join(item_image_root, image_name)] = mask_img_path
        elif "bg_removed" in image_names[1]:
            count[1] += 1

            for image_name in image_names:
                if "bg_removed" in image_name: continue
                mask_img_name = image_name.split(".")[0]+"_bg_removed.png"
                mask_img_path = os.path.join(item_image_root, mask_img_name)
                assert os.path.exists(mask_img_path), print(mask_img_path)
                image_to_mask_paths[os.path.join(item_image_root, image_name)] = mask_img_path
            
        if move_data and len(list(image_to_mask_paths.keys())):
            save_images_root = os.path.join(save_item_path, "images")
            
            os.makedirs(save_images_root, exist_ok=True)
            os.makedirs(save_masks_root, exist_ok=True)
            for img_path in image_to_mask_paths.keys():
                img_name = img_path.split("/")[-1]
                if not os.path.exists(os.path.join(save_images_root, img_name)):
                    os.symlink(img_path, os.path.join(save_images_root, img_name))
                if not os.path.exists(os.path.join(save_masks_root, img_name)):
                    os.symlink(image_to_mask_paths[img_path], os.path.join(save_masks_root, img_name))

            if not os.path.exists(os.path.join(save_item_path, "sparse")):
                os.symlink(os.path.join(item_root, "sparse"),os.path.join(save_item_path, "sparse"))

            with open(save_annotation_path, "w") as file:
                file.write(class_type+"\n")
                file.write(class_num+"\n")

            # exit()
    # exit()
    

print(count, "%.4f"%((count[0]+count[1])/count[2]))
