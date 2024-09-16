import os
import torch
import numpy as np

rgb_list = [[[],[],[]],[[],[],[]],[[],[],[]]]
root = "/path/to/your/gaussian-splatting/unigs/objaverse_all/objects"
for index,pkl_name in enumerate(os.listdir(root)):
    pkl_data = torch.load(os.path.join(root, pkl_name))
    guassians = pkl_data["3dgs"]
    
    for i in range(3):
        rgb_list[i][0].append( np.min(guassians[:,3+i]) )
        rgb_list[i][1].append( np.mean(guassians[:,3+i]) )
        rgb_list[i][2].append( np.max(guassians[:,3+i]) )
    
    print("\r %d | %d, R:[%.2f, %.2f, %.2f, %.2f, %.2f], G:[%.2f, %.2f, %.2f, %.2f, %.2f], B:[%.2f, %.2f, %.2f, %.2f, %.2f]"%
          (index, len(os.listdir(root)), 
        np.min(rgb_list[0][0]), np.mean(rgb_list[0][0]), np.mean(rgb_list[0][1]), np.mean(rgb_list[0][2]), np.max(rgb_list[0][2]),
        np.min(rgb_list[0][0]), np.mean(rgb_list[1][0]), np.mean(rgb_list[1][1]), np.mean(rgb_list[1][2]), np.max(rgb_list[1][2]),
        np.min(rgb_list[0][0]), np.mean(rgb_list[2][0]), np.mean(rgb_list[2][1]), np.mean(rgb_list[2][2]), np.max(rgb_list[2][2]),
        ))
