import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
import scipy.io as sio
from PIL import Image
from sunrgbd import SUN

# class SUN:
#     def __init__(self, 
#                 meta_file='//path/to/your/SUNRGBD/SUNRGBDMeta3DBB_v2.mat',
#                 meta_file_2D="//path/to/your/SUNRGBD/SUNRGBDMeta2DBB_v2.mat",
#                 rootPath='//path/to/your/SUNRGBD',
#                 using_fbx=False):
#         self.rootPath=rootPath
#         self.using_fbx = using_fbx
#         if not meta_file == None:
#             print('loading metadata into memory...')
#             tic = time.time()
#             self.dataSet = sio.loadmat(meta_file)['SUNRGBDMeta'].ravel()
#             self.dataSet2D = sio.loadmat(meta_file_2D)['SUNRGBDMeta2DBB'].ravel()
#             print('Done (t={:0.2f}s)'.format(time.time() - tic))
    
#     def getSensorDataId(self,sensorType='kv1'):
#         kv1Index=[]
#         for i in range(len(self.dataSet)):
#             if self.dataSet[i][8][0]==sensorType:
#                 kv1Index.append(i)
#         return kv1Index

#     def getPath(self,id):
#         sequenceName=self.dataSet[id][0][0]

#         if self.dataSet[id][8][0] == "kv2":
#             imgPath=os.path.join(self.rootPath,sequenceName,self.dataSet[id][4][0].split('/')[-2],self.dataSet[id][4][0].split('/')[-1])
#             depthPath=os.path.join(self.rootPath,sequenceName,self.dataSet[id][3][0].split('/')[-2],self.dataSet[id][3][0].split('/')[-1])
#             segPath=os.path.join(self.rootPath,sequenceName,'seg.mat')

#         else:
#             imgPath=os.path.join(self.rootPath,sequenceName,self.dataSet[id][4][0].split('//')[1])
#             depthPath=os.path.join(self.rootPath,sequenceName,self.dataSet[id][3][0].split('//')[1])
#             segPath=os.path.join(self.rootPath,sequenceName,'seg.mat')

#         return imgPath,depthPath,segPath

#     def load3dPoints(self, id):
#         """
#         read points from certain room
#         :param id: pos in metadata
#         :return: 3d points
#         """
#         data = self.dataSet[id]
#         sequenceName=data[0][0]
#         if self.dataSet[id][8][0] == "kv2":
#             depthPath=os.path.join(self.rootPath,sequenceName,data[3][0].split('/')[-2],data[3][0].split('/')[-1])
#         else:
#             depthPath=os.path.join(self.rootPath,sequenceName,data[3][0].split('//')[1])

#         if self.using_fbx: 
#             depthPath = "/".join([*depthPath.split("/")[:-2],"depth_bfx"])
#             depthPath = os.path.join(depthPath, os.listdir(depthPath)[0])

#         K=data[2]
#         Rtilt=data[1]
#         depthVis = Image.open(depthPath, 'r')
#         depthVisData = np.asarray(depthVis, np.uint16)
#         depthInpaint = np.bitwise_or(np.right_shift(depthVisData, 3), np.left_shift(depthVisData, 16 - 3))
#         depthInpaint = depthInpaint.astype(np.single) / 1000
#         depthInpaint[depthInpaint > 8] = 8
#         points3d= self.load3dPoints_(depthInpaint, K)
#         points3d = Rtilt.dot(points3d.T).T
#         return points3d, depthInpaint

#     def load3dPoints_(self, depth, K):
#         cx, cy = K[0, 2], K[1, 2]
#         fx, fy = K[0, 0], K[1, 1]
#         invalid = depth == 0
#         x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
#         xw = (x - cx) * depth / fx
#         yw = (y - cy) * depth / fy
#         zw = depth
#         points3dMatrix = np.stack((xw, zw, -yw), axis=2)
#         points3dMatrix[np.stack((invalid, invalid, invalid), axis=2)] = np.nan
#         points3d = points3dMatrix.reshape(-1, 3)
#         return points3d

#     def visPointCloud(self, id):
#         import open3d as o3d
#         points3d, depth = self.load3dPoints(id)
#         # plt.title('depth')
#         # plt.imshow(depth)
#         # plt.show()
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points3d)
#         o3d.visualization.draw_geometries([pcd])

#     def getImg(self,id,only_img=False):
#         img,depth,segLabel,segInstances = None, None, None, None
#         imgPath,depthPath,segPath=self.getPath(id)
#         img=plt.imread(imgPath)
#         if only_img: return img

#         depth=plt.imread(depthPath)
#         seg= sio.loadmat(segPath)
#         if "seglabel" in seg.keys(): segLabel=seg['seglabel']
#         if "seginstances" in seg.keys(): segInstances=seg['seginstances']
#         return img,depth,segLabel,segInstances

#     def visImg(self, id):
#         img,depth,segl,segi=self.getImg(id)
#         plt.subplot(2,2,1)
#         plt.imshow(img)
#         plt.title("img")

#         plt.subplot(2,2,2)
#         plt.imshow(depth)
#         plt.title("depth")

#         plt.subplot(2,2,3)
#         plt.imshow(segl)
#         plt.title("seglabel")

#         plt.subplot(2,2,4)
#         plt.imshow(segi)
#         plt.title("seginstances")

#         plt.savefig("/path/to/your/SUN-RGB-D/img.png")

#     def getCornerList(self,id):
#         cornerList, classNameList = [], []

#         data=self.dataSet[id][10].flatten()
#         for i in range(len(data)):
#             basis=data[i][0]
#             coeffs=data[i][1][0]
#             centroid=data[i][2]
#             className=data[i][3][0]
#             label=data[i][6]
#             corner=self.getCorner(basis,coeffs,centroid)

#             cornerList.append(corner)
#             classNameList.append(className)
#         return cornerList, classNameList

#     def flip_toward_viewer(self,normals, points):
#         points /= np.linalg.norm(points, axis=1)
#         projection = np.sum(points * normals, axis=1)
#         flip = projection > 0
#         normals[flip] = - normals[flip]
#         return normals

#     def getCorner(self,basis,coeffs,centroid):
#         corner = np.zeros((8, 3), dtype=np.float32)
#         coeffs = coeffs.ravel()
#         indices = np.argsort(- np.abs(basis[:, 0]))
#         basis = basis[indices, :]
#         coeffs = coeffs[indices]
#         indices = np.argsort(- np.abs(basis[1:3, 1]))
#         if indices[0] == 1:
#             basis[[1, 2], :] = basis[[2, 1], :]
#             coeffs[[1, 2]] = coeffs[[2, 1]]

#         basis = self.flip_toward_viewer(basis, np.repeat(centroid, 3, axis=0))
#         coeffs = abs(coeffs)
#         corner[0] = -basis[0] * coeffs[0] + basis[1] * coeffs[1] + basis[2] * coeffs[2]
#         corner[1] = basis[0] * coeffs[0] + basis[1] * coeffs[1] + basis[2] * coeffs[2]
#         corner[2] = basis[0] * coeffs[0] + -basis[1] * coeffs[1] + basis[2] * coeffs[2]
#         corner[3] = -basis[0] * coeffs[0] + -basis[1] * coeffs[1] + basis[2] * coeffs[2]

#         corner[4] = -basis[0] * coeffs[0] + basis[1] * coeffs[1] + -basis[2] * coeffs[2]
#         corner[5] = basis[0] * coeffs[0] + basis[1] * coeffs[1] + -basis[2] * coeffs[2]
#         corner[6] = basis[0] * coeffs[0] + -basis[1] * coeffs[1] + -basis[2] * coeffs[2]
#         corner[7] = -basis[0] * coeffs[0] + -basis[1] * coeffs[1] + -basis[2] * coeffs[2]
#         corner += np.repeat(centroid, 8, axis=0)
#         return corner

#     def visCube(self,id,m=0,length=100):
#         import open3d as o3d
#         cornerList, classNameList =self.getCornerList(id)
#         lines = [[0, 1],[1, 2],[2, 3],[3, 0],[0, 4],[1, 5],[2, 6],
#             [3, 7],[4, 5],[5, 6],[6, 7],[7, 4]]
#         colors = [[0, 0, 1] for i in range(len(lines))]
#         ll=[]
#         length = min(length, len(cornerList))
#         for i in range(length):
#             line_set = o3d.geometry.LineSet(
#                 points=o3d.utility.Vector3dVector(cornerList[i]),
#                 lines=o3d.utility.Vector2iVector(lines),
#             )
#             line_set.colors = o3d.utility.Vector3dVector(colors)
#             ll.append(line_set)

#         line_set.colors = o3d.utility.Vector3dVector(colors)

#         # X-axis : 红色箭头
#         # Y-axis : 绿色箭头
#         # Z-axis : 蓝色箭头
#         coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
#         ll.append(coord_frame)

#         if m==0:
#             o3d.visualization.draw_geometries(ll)
#         elif m==1:
#             points3d, depth = self.load3dPoints(id)
#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(points3d)
#             ll.append(pcd)   
#             o3d.visualization.draw_geometries(ll)
    
#     def getCornerList2D(self,id):
#         cornerList, classNameList = [], []

#         data=self.dataSet2D[id][1].flatten()
#         for i in range(len(data)):
#             objid=data[i][0][0]
#             gtBb2D=data[i][1][0]
#             className=data[i][2][0]
#             has3dbox=data[i][3][0]

#             cornerList.append(gtBb2D)
#             classNameList.append(className)
#         return cornerList, classNameList

#     def visCube2D(self,id,m=0,img=None,length=100):
#         cornerList, classNameList =self.getCornerList2D(id)

#         if img is None:
#             img,depth,segl,segi=self.getImg(id)

#         if m==1: 
#             plt.imshow(img)

#         for i,bbox in enumerate(cornerList):
#             if bbox[2] < 10 or bbox[3]<10: continue

#             color = "%x"%(np.random.randint(256)*np.random.randint(256)*np.random.randint(256))
#             if len(color) == 5: color = "0"+color
#             color = "#" + color.upper()
            
#             plt.text(bbox[0], bbox[1], classNameList[i], color=color)
#             rectangle = plt.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3], fill=False, color=color)
#             plt.gca().add_patch(rectangle)
#         plt.savefig("/path/to/your/SUN-RGB-D/visCube2D.png")

# dataset item
meta_file_path = "//path/to/your/SUNRGBD/SUNRGBDMeta3DBB_v2.mat"
meta_file_2D_path = "//path/to/your/SUNRGBD/SUNRGBDMeta2DBB_v2.mat"
root = "//path/to/your/SUNRGBD"
sun_data = SUN(meta_file_path, meta_file_2D_path, root, using_fbx=True)

output_root = "/path/to/your/gaussian-splatting/output"

# get item id 
kv1Index=sun_data.getSensorDataId('kv1')
kv2Index=sun_data.getSensorDataId('kv2')
realsenseIndex=sun_data.getSensorDataId('realsense')
xtionIndex=sun_data.getSensorDataId('xtion')
allIndex = [*kv1Index, *kv2Index, *realsenseIndex, *xtionIndex] # 10335

train_ave_psnr = []
train_ave_l1 = []

for item_id in tqdm(allIndex):
    # root path
    sequenceName=sun_data.dataSet[item_id][0][0]
    item_path = os.path.join(output_root, sequenceName)
    gs3d_path = os.path.join(item_path, "point_cloud", "iteration_500", "point_cloud.ply")
    json_path = os.path.join(item_path, "training_results.json")
    if not (os.path.exists(item_path) and os.path.exists(gs3d_path)):
        print(os.path.exists(item_path), os.path.exists(gs3d_path))
        print(sequenceName)
        print()
    
    # print(json_path)
    # exit()
    with open(json_path, "r") as file:
        content = json.load(file)
        train_ave_psnr.append(content["train"]["psnr"])
        train_ave_l1.append(content["train"]["l1"])

print("PSNR: %.4f"%( sum(train_ave_psnr)/len(train_ave_psnr) ))
print("L1: %.4f"%( sum(train_ave_l1)/len(train_ave_l1) ))