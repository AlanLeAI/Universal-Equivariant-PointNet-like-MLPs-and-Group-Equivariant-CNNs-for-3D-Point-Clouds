import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import torch.nn.functional as F
import math
import numpy as np
from utils.utils import show_point_cloud
import sys
import os
sys.path.insert(0,'./')
from data_utils.ModelNetDataLoader import ModelNetDataLoader

DATA_PATH = './data/modelnet40_normal_resampled/'

# TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='train', normal_channel=True)
# trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=32, shuffle=True, num_workers=16)

# points, labels = iter(trainDataLoader)



# def main():
#     print(points.shape, labels.points, sep='-->')

import parallelTestModule

if __name__ == '__main__':    
    extractor = parallelTestModule.ParallelExtractor()
    extractor.runInParallel(numProcesses=2, numThreads=4)
