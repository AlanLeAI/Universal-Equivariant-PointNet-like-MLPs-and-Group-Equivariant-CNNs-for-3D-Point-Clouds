"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils.pointconv_util import RotationPointConvDensitySetAbstraction

class GroupPointConvDensityClsSsg(nn.Module):
    def __init__(self, num_classes = 40, split_group=24):
        super(GroupPointConvDensityClsSsg, self).__init__()
        feature_dim = 3
        npoint_dim = 16

        self.sa1 = RotationPointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=feature_dim + npoint_dim, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False, split_group=split_group)
        # add residual blocks
        self.sa2 = RotationPointConvDensitySetAbstraction(npoint=128, nsample=32, in_channel=128 + npoint_dim, mlp=[128, 128, 256], bandwidth = 0.2, group_all=False, split_group=split_group)
        # add residual blocks
        self.sa3 = RotationPointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=256 + npoint_dim, mlp=[256, 512, 1024], bandwidth = 0.4, group_all=True, split_group=split_group)
        # add residual blocks

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.7)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # x = l3_points.view(B, 1024)
        # x = self.drop1(F.relu(self.fc1(x)))
        # x = self.drop2(F.relu(self.fc2(x)))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)

        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,6,2048))
    label = torch.randn(8,16)
    xyz = input[:,:3,2048]
    feat = input[:3,:,2048]
    model = GroupPointConvDensityClsSsg(num_classes=40)
    output= model(xyz,feat)
    print(output.size())

