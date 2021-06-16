# GPointConv
## Installation
The code is modified from repo https://github.com/DylanWusee/pointconv_pytorch

## Tensorflow version
https://github.com/kenakai16/gpointconv_tf

## Usage
### ModelNet40 Classification

Download the ModelNet40 dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip). This dataset is the same one used in [PointNet](https://arxiv.org/abs/1612.00593), thanks to [Charles Qi](https://github.com/charlesq34/pointnet).

To train the model,
```
python train_cls_gconv.py --model gpointconv_modelnet40 --normal
```

To evaluate the model,
```
python eval_cls_gconv.py --checkpoint ./checkpoint/check_point.pth --normal
```
We are updating the latest weight...
## License
This repository is released under MIT License (see LICENSE file for details).



# Universal-Equivariant-PointNet-like-MLPs-and-Group-Equivariant-CNNs-for-3D-Point-Clouds
