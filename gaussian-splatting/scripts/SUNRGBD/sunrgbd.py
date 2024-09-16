import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image
import open3d as o3d
from tqdm import tqdm

# https://github.com/ankurhanda/SceneNetv1.0
class SUN:
    def __init__(self, 
                meta_file='//path/to/your/SUNRGBD/SUNRGBDMeta3DBB_v2.mat',
                meta_file_2D="//path/to/your/SUNRGBD/SUNRGBDMeta2DBB_v2.mat",
                rootPath='//path/to/your/SUNRGBD',
                using_fbx=False):
        self.rootPath=rootPath
        self.using_fbx = using_fbx
        if not meta_file == None:
            print('loading metadata into memory...')
            tic = time.time()
            self.dataSet = sio.loadmat(meta_file)['SUNRGBDMeta'].ravel()
            self.dataSet2D = sio.loadmat(meta_file_2D)['SUNRGBDMeta2DBB'].ravel()
            splitSet = sio.loadmat(os.path.join(rootPath, "SUNRGBDtoolbox", "traintestSUNRGBD", "allsplit.mat"))
            self.splitSet = {
                "alltrain": splitSet["alltrain"].ravel(), # 5285
                "alltest": splitSet["alltest"].ravel(), # 5050
                "train": np.reshape(splitSet["trainvalsplit"].ravel()[0][0], (-1)), # 2666
                "val": np.reshape(splitSet["trainvalsplit"].ravel()[0][1], (-1)), #2619
            }
            print('SUN Loading meta data Done (t={:0.2f}s)'.format(time.time() - tic))
        
        # with open("//path/to/your/SUNRGBD/scripts/trash.txt", "w") as file:
        #     sequence_names = np.concatenate((self.splitSet["alltrain"],
        #                                      self.splitSet["alltest"],
        #                                      self.splitSet["train"],
        #                                      self.splitSet["val"]),axis=0)
        #     for sequence_name in sequence_names:
        #         file.write(sequence_name[0]+"\n")
    
    def search_id_by_name(self, sequence_name):
        for i in range(len(self.dataSet)):
            if sequence_name == self.dataSet[i][0][0]:
                return i
        return -1
    
    def get_split_json(self):
        data_split_json = {
            "alltrain":[],
            "alltest":[],
            "train":[],
            "val":[],
        }
        data_type_dict = {"alltrain":"train","alltest":"test","train":"train","val":"val"}
        data_to_split = {}
        for key in self.splitSet.keys():
            for ori_sequence_name in self.splitSet[key]:
                sequence_name:str = ori_sequence_name[0][17:]
                if sequence_name.endswith("/"): sequence_name = sequence_name[:-1]
                data_split_json[key].append(sequence_name) # /n/fs/sun3d/data/
                data_to_split[sequence_name] = data_type_dict[key]
        return data_split_json, data_to_split
    
    def getSensorDataId(self,sensorType='kv1'):
        kv1Index=[]
        for i in range(len(self.dataSet)):
            if self.dataSet[i][8][0]==sensorType:
                kv1Index.append(i)
        return kv1Index

    def getPath(self,id):
        sequenceName=self.dataSet[id][0][0]

        if self.dataSet[id][8][0] == "kv2":
            imgPath=os.path.join(self.rootPath,sequenceName,self.dataSet[id][4][0].split('/')[-2],self.dataSet[id][4][0].split('/')[-1])
            depthPath=os.path.join(self.rootPath,sequenceName,self.dataSet[id][3][0].split('/')[-2],self.dataSet[id][3][0].split('/')[-1])
            segPath=os.path.join(self.rootPath,sequenceName,'seg.mat')

        else:
            imgPath=os.path.join(self.rootPath,sequenceName,self.dataSet[id][4][0].split('//')[1])
            depthPath=os.path.join(self.rootPath,sequenceName,self.dataSet[id][3][0].split('//')[1])
            segPath=os.path.join(self.rootPath,sequenceName,'seg.mat')

        return imgPath,depthPath,segPath

    def load3dPoints(self, id):
        """
        read points from certain room
        :param id: pos in metadata
        :return: 3d points
        """
        data = self.dataSet[id]
        sequenceName=data[0][0]
        if self.dataSet[id][8][0] == "kv2":
            depthPath=os.path.join(self.rootPath,sequenceName,data[3][0].split('/')[-2],data[3][0].split('/')[-1])
        else:
            depthPath=os.path.join(self.rootPath,sequenceName,data[3][0].split('//')[1])

        if self.using_fbx: 
            depthPath = "/".join([*depthPath.split("/")[:-2],"depth_bfx"])
            depthPath = os.path.join(depthPath, os.listdir(depthPath)[0])

        K=data[2]
        Rtilt=data[1]
        depthVis = Image.open(depthPath, 'r')
        depthVisData = np.asarray(depthVis, np.uint16)
        depthInpaint = np.bitwise_or(np.right_shift(depthVisData, 3), np.left_shift(depthVisData, 16 - 3))
        depthInpaint = depthInpaint.astype(np.single) / 1000
        depthInpaint[depthInpaint > 8] = 8
        points3d= self.load3dPoints_(depthInpaint, K)
        points3d = Rtilt.dot(points3d.T).T
        return points3d, depthInpaint

    def load3dPoints_(self, depth, K):
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]
        invalid = depth == 0
        x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        xw = (x - cx) * depth / fx
        yw = (y - cy) * depth / fy
        zw = depth
        points3dMatrix = np.stack((xw, zw, -yw), axis=2)
        points3dMatrix[np.stack((invalid, invalid, invalid), axis=2)] = np.nan
        points3d = points3dMatrix.reshape(-1, 3)
        return points3d

    def visPointCloud(self, id):
        points3d, depth = self.load3dPoints(id)
        # plt.title('depth')
        # plt.imshow(depth)
        # plt.show()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points3d)
        o3d.visualization.draw_geometries([pcd])

    def getImg(self,id,only_img=False):
        img,depth,segLabel,segInstances = None, None, None, None
        imgPath,depthPath,segPath=self.getPath(id)
        img=plt.imread(imgPath)
        if only_img: return img

        depth=plt.imread(depthPath)
        seg= sio.loadmat(segPath)
        if "seglabel" in seg.keys(): segLabel=seg['seglabel']
        if "seginstances" in seg.keys(): segInstances=seg['seginstances']
        return img,depth,segLabel,segInstances

    def visImg(self, id):
        img,depth,segl,segi=self.getImg(id)
        plt.subplot(2,2,1)
        plt.imshow(img)
        plt.title("img")

        plt.subplot(2,2,2)
        plt.imshow(depth)
        plt.title("depth")

        plt.subplot(2,2,3)
        plt.imshow(segl)
        plt.title("seglabel")

        plt.subplot(2,2,4)
        plt.imshow(segi)
        plt.title("seginstances")

        plt.savefig("/path/to/your/SUN-RGB-D/img.png")

    def getCornerList(self,id):
        cornerList, classNameList = [], []

        data=self.dataSet[id][10].flatten()
        for i in range(len(data)):
            basis=data[i][0]
            coeffs=data[i][1][0]
            centroid=data[i][2]
            className=data[i][3]
            label=data[i][6]
            corner=self.getCorner(basis,coeffs,centroid)

            cornerList.append(corner)
            classNameList.append(className)
        return cornerList, classNameList

    def flip_toward_viewer(self,normals, points):
        points /= np.linalg.norm(points, axis=1)
        projection = np.sum(points * normals, axis=1)
        flip = projection > 0
        normals[flip] = - normals[flip]
        return normals

    def getCorner(self,basis,coeffs,centroid):
        corner = np.zeros((8, 3), dtype=np.float32)
        coeffs = coeffs.ravel()
        indices = np.argsort(- np.abs(basis[:, 0]))
        basis = basis[indices, :]
        coeffs = coeffs[indices]
        indices = np.argsort(- np.abs(basis[1:3, 1]))
        if indices[0] == 1:
            basis[[1, 2], :] = basis[[2, 1], :]
            coeffs[[1, 2]] = coeffs[[2, 1]]

        basis = self.flip_toward_viewer(basis, np.repeat(centroid, 3, axis=0))
        coeffs = abs(coeffs)
        corner[0] = -basis[0] * coeffs[0] + basis[1] * coeffs[1] + basis[2] * coeffs[2]
        corner[1] = basis[0] * coeffs[0] + basis[1] * coeffs[1] + basis[2] * coeffs[2]
        corner[2] = basis[0] * coeffs[0] + -basis[1] * coeffs[1] + basis[2] * coeffs[2]
        corner[3] = -basis[0] * coeffs[0] + -basis[1] * coeffs[1] + basis[2] * coeffs[2]

        corner[4] = -basis[0] * coeffs[0] + basis[1] * coeffs[1] + -basis[2] * coeffs[2]
        corner[5] = basis[0] * coeffs[0] + basis[1] * coeffs[1] + -basis[2] * coeffs[2]
        corner[6] = basis[0] * coeffs[0] + -basis[1] * coeffs[1] + -basis[2] * coeffs[2]
        corner[7] = -basis[0] * coeffs[0] + -basis[1] * coeffs[1] + -basis[2] * coeffs[2]
        corner += np.repeat(centroid, 8, axis=0)
        return corner

    def visCube(self,id,m=0,length=100):
        cornerList, classNameList =self.getCornerList(id)
        lines = [[0, 1],[1, 2],[2, 3],[3, 0],[0, 4],[1, 5],[2, 6],
            [3, 7],[4, 5],[5, 6],[6, 7],[7, 4]]
        colors = [[0, 0, 1] for i in range(len(lines))]
        ll=[]
        length = min(length, len(cornerList))
        for i in range(length):
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(cornerList[i]),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            ll.append(line_set)

        line_set.colors = o3d.utility.Vector3dVector(colors)

        # X-axis : 红色箭头
        # Y-axis : 绿色箭头
        # Z-axis : 蓝色箭头
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        ll.append(coord_frame)

        if m==0:
            o3d.visualization.draw_geometries(ll)
        elif m==1:
            points3d, depth = self.load3dPoints(id)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points3d)
            ll.append(pcd)   
            o3d.visualization.draw_geometries(ll)
    
    def getCornerList2D(self,id,ensure_3d=False):
        cornerList, classNameList = [], []

        data=self.dataSet2D[id][1].flatten()
        for i in range(len(data)):
            objid=data[i][0][0]
            gtBb2D=data[i][1][0]
            className=data[i][2][0]
            has3dbox=data[i][3][0]

            if ensure_3d and has3dbox[0] == 0:
                continue

            cornerList.append(gtBb2D)
            classNameList.append(className)
        return cornerList, classNameList

    def visCube2D(self,id,m=0,img=None,length=100):
        cornerList, classNameList =self.getCornerList2D(id)

        if img is None:
            img,depth,segl,segi=self.getImg(id)

        if m==1: 
            plt.imshow(img)

        for i,bbox in enumerate(cornerList):
            if bbox[2] < 10 or bbox[3]<10: continue

            color = "%x"%(np.random.randint(256)*np.random.randint(256)*np.random.randint(256))
            if len(color) == 5: color = "0"+color
            color = "#" + color.upper()
            
            plt.text(bbox[0], bbox[1], classNameList[i], color=color)
            rectangle = plt.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3], fill=False, color=color)
            plt.gca().add_patch(rectangle)
        plt.savefig("/path/to/your/SUN-RGB-D/visCube2D.png")

if __name__ == "__main__":
    meta_file_path = "//path/to/your/SUNRGBD/SUNRGBDMeta3DBB_v2.mat"
    meta_file_2D_path = "//path/to/your/SUNRGBD/SUNRGBDMeta2DBB_v2.mat"
    root = "//path/to/your/SUNRGBD"

    sun = SUN()
    kv1Index=sun.getSensorDataId('kv1')
    kv2Index=sun.getSensorDataId('kv2')
    realsenseIndex=sun.getSensorDataId('realsense')
    xtionIndex=sun.getSensorDataId('xtion')
    allIndex = [*kv1Index, *kv2Index, *realsenseIndex, *xtionIndex] # 10335

    label_num = {}
    for id in tqdm(allIndex):

        # imgs
        img,depth,segl,segi=sun.getImg(id)

        # pcd
        points3d, _ = sun.load3dPoints(id) # [x,y,z]

        # bbox
        cornerList3D, classNameList3D =sun.getCornerList(id)
        cornerList2D, classNameList2D =sun.getCornerList2D(id,ensure_3d=False)

        # camera pose
        sequenceName=sun.dataSet[id][0][0]
        item_path = os.path.join(root, sequenceName)
        if os.path.exists(os.path.join(item_path, "fullres", "intrinsics.txt")):
            intrinsics = np.loadtxt(os.path.join(item_path, "fullres", "intrinsics.txt"))
        else: intrinsics = np.loadtxt(os.path.join(item_path, "intrinsics.txt"))
        intrinsics = np.reshape(intrinsics,(-1))

        extrinsics_root = os.path.join(item_path, "extrinsics")
        extrinsics_name = os.listdir(extrinsics_root)[-1]
        extrinsics = np.loadtxt(os.path.join(extrinsics_root, extrinsics_name))

        transform_matrix = np.array([[1, 0, 0],
                                    [0, 0, -1],
                                    [0, 1, 0]])
        extrinsics[:3,:3] = np.matmul(extrinsics[:3,:3], transform_matrix)
        extrinsics = extrinsics.tolist()
        extrinsics.append([0,0,0,1])
        exit()

'''SUNRGBDMeta
[0]: ['SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize']

[1]: 
[
    (
        array([[ 0.99221595, -0.12452913,  0.        ],[ 0.12452913,  0.99221595,  0.        ],[ 0.        ,  0.        ,  1.        ]]), 
        array([[0.94379202, 1.15103499, 0.98480693]]), 
        array([[ 1.04730715,  4.16869579, -0.24685933]]), 
        array(['bed'], dtype='<U3'), 
        array([], dtype='<U1'), 
        array(['SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize'],dtype='<U81'), 
        array([[-0.12452913, -0.99221595,  0.        ]]), 
        array([[328, 152, 346, 320]], dtype=uint16), 
        array([[1]], dtype=uint8)
        )
    (
        array([[ 0.99686539, -0.0791163 ,  0.        ],[ 0.08304548,  0.99654576,  0.        ],[ 0.        ,  0.        ,  1.        ]]), 
        array([[0.2872641, 0.2736726, 0.45     ]]), 
        array([[ 2.43181818,  4.84090909, -0.75      ]]), 
        array(['night_stand'], dtype='<U11'), 
        array([], dtype='<U1'), 
        array(['SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize'],dtype='<U81'), 
        array([[-0.08304548, -0.99654576,  0.        ]]), 
        array([[568.53840816, 276.61940951,  95.35192217,  92.26649057]]), 
        array([[1]], dtype=uint8))
    (
        array([[ 0.99409097, -0.10855016,  0.        ],[ 0.10333986,  0.99464611,  0.        ],[ 0.        ,  0.        ,  1.        ]]), 
        array([[0.79561037, 0.35188395, 0.33863636]]), 
        array([[ 0.82272727,  2.51818182, -0.86136364]]), 
        array(['ottoman'], dtype='<U7'), 
        array([], dtype='<U1'), 
        array(['SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize'],dtype='<U81'), 
        array([[-0.10333986, -0.99464611,  0.        ]]), 
        array([[324, 319, 325, 211]], dtype=uint16), 
        array([[1]], dtype=uint8))
    (
        array([[ 0.11196753,  0.99371187,  0.        ],[-0.99442258,  0.10546906,  0.        ],[ 0.        ,  0.        ,  1.        ]]), 
        array([[0.9743084 , 0.30168261, 1.02045455]]), 
        array([[-0.83181818,  3.84545455, -0.17954545]]), 
        array(['dresser_mirror'], dtype='<U14'), 
        array([], dtype='<U1'), 
        array(['SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize'],dtype='<U81'), 
        array([[ 0.99442258, -0.10546906,  0.        ]]), 
        array([[154,  17, 151, 368]], dtype=uint16), 
        array([[1]], dtype=uint8))
    (
        array([[ 0.10468478,  0.99450545,  0.        ],[-0.99478144,  0.10202887,  0.        ],[ 0.        ,  0.        ,  1.        ]]), 
        array([[0.78156708, 0.35640538, 1.05227273]]), 
        array([[-1.08636364,  1.90454545, -0.14772727]]), 
        array(['dresser'], dtype='<U7'), 
        array([], dtype='<U1'), 
        array(['SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize'],dtype='<U81'), 
        array([[ 0.99478144, -0.10202887,  0.        ]]), 
        array([[  1,   1, 221, 529]], dtype=uint16), 
        array([[1]], dtype=uint8)
        )

[2]:[[ 0.979589  0.012593 -0.200614]
 [ 0.012593  0.992231  0.123772]
 [ 0.200614 -0.123772  0.97182 ]]

[3]: [[529.5   0.  365. ]
 [  0.  529.5 265. ]
 [  0.    0.    1. ]]

[4]: ['/n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/depth/0000103.png']

[5]: ['/n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/image/0000103.jpg']

[6]: [[ 0.979589  0.200614  0.012593]
 [-0.200614  0.97182   0.123772]
 [ 0.012593 -0.123772  0.992231]]

[7]: ['0000103.png']

[8]: ['0000103.jpg']

[9]: ['kv2']

[10]: [[1]]

[11]: [-0.03000046 -1.56999899 -0.92999546  3.43000047  4.88000432  8.8500011
8.689994   -0.0299999  -1.56999843 -0.9299949   3.43000102  4.88000488
   8.85000165  8.68999456]
 [-0.05999994  0.19999904  5.740002    5.12000492 12.02000908 12.12001204
   2.26000749 -0.06000008  0.19999891  5.74000187  5.12000479 12.02000895
  12.1200119   2.26000736]
 [ 4.82999806  4.8299982   4.82999826  4.82999786  4.82999788  4.82999752
   4.82999732 -1.28999948 -1.28999933 -1.28999927 -1.28999968 -1.28999966
  -1.29000002 -1.29000022]]

[12]: [[(array([[328, 152, 346, 320]], dtype=uint16),)
  (array([[568.53840816, 276.61940951,  95.35192217,  92.26649057]]),)
  (array([[324, 319, 325, 211]], dtype=uint16),)
  (array([[154,  17, 151, 368]], dtype=uint16),)
  (array([[  1,   1, 221, 529]], dtype=uint16),)]]

'''

'''groudtruth
[
    (
        array([[ 0.99221595, -0.12452913,  0.        ],[ 0.12452913,  0.99221595,  0.        ],[ 0.        ,  0.        ,  1.        ]]), 
        array([[0.94379202, 1.15103499, 0.98480693]]), 
        array([[ 1.04730715,  4.16869579, -0.24685933]]), 
        array(['bed'], dtype='<U3'), array([], dtype='<U1'), 
        array(['SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize'],dtype='<U81'), 
        array([[-0.12452913, -0.99221595,  0.        ]]), 
        array([[328, 152, 346, 320]], dtype=uint16), 
        array([[1]], dtype=uint8)
        )
    (
        array([[ 0.99686539, -0.0791163 ,  0.        ],[ 0.08304548,  0.99654576,  0.        ],[ 0.        ,  0.        ,  1.        ]]), 
        array([[0.2872641, 0.2736726, 0.45     ]]), 
        array([[ 2.43181818,  4.84090909, -0.75      ]]), 
        array(['night_stand'], dtype='<U11'), array([], dtype='<U1'), 
        array(['SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize'],dtype='<U81'), 
        array([[-0.08304548, -0.99654576,  0.        ]]), 
        array([[568.53840816, 276.61940951,  95.35192217,  92.26649057]]), 
        array([[1]], dtype=uint8)
        )
    (
        array([[ 0.99409097, -0.10855016,  0.        ],[ 0.10333986,  0.99464611,  0.        ],[ 0.        ,  0.        ,  1.        ]]), 
        array([[0.79561037, 0.35188395, 0.33863636]]), 
        array([[ 0.82272727,  2.51818182, -0.86136364]]), 
        array(['ottoman'], dtype='<U7'), array([], dtype='<U1'), 
        array(['SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize'],dtype='<U81'), 
        array([[-0.10333986, -0.99464611,  0.        ]]), 
        array([[324, 319, 325, 211]], dtype=uint16), 
        array([[1]], dtype=uint8)
        )
]
'''
