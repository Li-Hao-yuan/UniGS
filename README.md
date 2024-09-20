## UniGS: Unified Language-Image-3D Pretraining with Gaussian Splatting

- Paper: [Arxiv](xxxxx)
- Dataset: [Huggingface](https://huggingface.co/datasets/lihy285/UniGS)


## Install
- Install pytorch packages
```Shell
conda create -n unigs python==3.8.19
conda activate unigs
pip install -r requirements.txt
```
Details of python environments are provided in pip_CUDA11.4.txt and pip_CUDA12.4.txt.

- (Optional) Install xformers
```Shell
## torch==2.3.0
pip install xformers

## torch<2.3.0 with cuda==11.4 (compiling takes up to 30 min)
export CUDA_HOME=/usr/local/cuda-11.4
conda install -c nvidia/label/cuda-11.4.0 cuda-nvcc
conda install -c conda-forge gcc
conda install -c conda-forge gxx_linux-64

git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .
```

## Dataset
We have open sourced Objaverse objects with 1024 and 10000 3D gaussians on [Huggingface](https://huggingface.co/datasets/lihy285/UniGS). You can choose to download the processed data, or process your data:
```Shell
cd /Users/lhy/Desktop/vs/workspace/UniGS/gaussian-splatting/scripts

# choose one dataset, for example objaverse
cd Objaverse

# first conver the data format to 3DGS
python convert_objaverse_to_3DGS.py 

# then generated 3DGS processing command
python sample_objaverse.py

# and run then for processed 3DGS
cd ..
python running_sh.py

# finally convert 3DGS format to uings training data
cd Objaverse
python make_datasets.py

```
All data processing maintains this order, and there will be changes according to different datasets. For example, MVImgNet is saved in COLMAP format, thus can skip the process of converting data format. Meanwhile, the optimization commands for different data sets are different in "sample_\<dataset>.py".

We thanks [Zero123](https://github.com/cvlab-columbia/zero123) for [the rendered images of objaverse, you can download it from the **Dataset (Objaverse Renderings)** part of Zero123.


## Training
The model training command is as follows

```Shell
python train.py config/objaverse/objaverse_unigs_classification.py --gpu-ids 0 1 2 3 4 5 
```

## Evaling
The evaluating command is as follows
```Shell
python ./lib/eval.py \
    config/objaverse/objaverse_unigs.py  \
    --resume-from /path/to/your/model.pth \
    --task retrive \
    --dataset objaverse \
    --test_dataset objaverse \
    --data_dir ./gaussian-splatting/objaverse \
    --data_split sample_all.json \
    --batch_size 40 \
    --k 1,3,5
```

## Results
__Table. Summary of the experimental results on Objaverse-LVIS zero-shot classification.__
>
>**Default-scale training**
>
| Methods | Source   |  3D points   | Backbone | Avg. Top1 | Avg. Top3 | Avg. Top 5 | Dataset train| Dataset test| Representation   |
|---------|-----------|------------------------|-----------|-----------|-----------|-------------|-------------------|-------------------|-------------------|
| Uni3D | no LVIS      | 1024       | EVA02-S-patch14      | 36.72     | 57.09     | 65.18       | 100k      | 46k | point clouds
| Uni3D   |           |                        |           | 30.47     | 48.46     | 55.87       |       |  | 3DGS              |
| TAMM    |           |                        |           | 22.70     | 38.83     | 47.13       | ||3DGS              |
| ReCon   |           |                        |           | 23.40     | 41.41     | 48.95       | ||3DGS              |
| UniGS   |           |                        |           | **38.57** | **60.57** | **68.96**   | ||3DGS              |
>
>**Large-scale training**
>
| Methods | Source   |  3D points   | Backbone | Avg. Top1 | Avg. Top3 | Avg. Top 5 | Dataset train| Dataset test| Representation   |
|---------|-----------|------------------------|-----------|-----------|-----------|-------------|-------------------|-------------------|-------------------|
|Uni3D| with LVIS  |  1024   | EVA02-S-patch14      | 46.31     | 72.62     | 79.78       |  800k      | 46k | point clouds
| UniGS   |           |                        |           | **49.95** | **75.60** | **82.38**   |    |  | 3DGS

>
>**10000 3D points**
>
| Methods | Source   |  3D points   | Backbone | Avg. Top1 | Avg. Top3 | Avg. Top 5 | Dataset train| Dataset test| Representation   |
|---------|-----------|------------------------|-----------|-----------|-----------|-------------|-------------------|-------------------|-------------------|
| TAMM（**Offical model**） | with LVIS |  **10000** | Point-BERT | 50.70 | 73.20 | 80.60 | 800k | 46k | point clouds |
| ReCon++(**Offical model**） | |  | ViT-bigG-patch14 | 53.20 | 75.30 | 81.50 | 800k | 46k | point clouds |
| Uni3D（**Offical model**） | |  | EVA02-S-patch14 | 50.34 | 72.70 | 79.81 | 800k | 46k | point clouds |
| Uni3D（**Offical model**） | |  |EVA02-S-patch14  | 49.87 | 72.39 | 79.70 | 800k | 6k | point clouds |
| UniGS     |      |     |    EVA02-S-patch14    | 51.22 | 73.64 | 80.88   | 46k | 6k| 3DGS |

## TODO
We will supplement experiments on large-scale Objaverse dataset with 10000 3D gaussians.

## Acknowledgement
Uni3D is built using the awesome [EVA](https://github.com/baaivision/EVA), [OpenCLIP](https://github.com/mlfoundations/open_clip), [timm](https://github.com/huggingface/pytorch-image-models/), [OpenShape](https://github.com/Colin97/OpenShape_code) and [Uni3D](https://github.com/baaivision/Uni3D).