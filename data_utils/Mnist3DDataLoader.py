import numpy as np
import warnings
import os
import torch
import h5py
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class Mnist3D_H5Dataset(torch.utils.data.Dataset):

    def __init__(self, file_path):
        super(Mnist3D_H5Dataset, self).__init__()
        h5_file = h5py.File(file_path, 'r')
        self.mnist_imgs = []
        self.mnist_points = []
        self.mnist_targets = []
        self.mnist_index = []
        for i in range(len(h5_file)):
            d = h5_file[str(i)]
            self.mnist_points.append(np.array(d["points"][:]))
            self.mnist_targets.append(d.attrs["label"])
            self.mnist_imgs.append(np.array(d["img"][:]))

    def _get_item_(self, index):
        self.mnist_points[index] = farthest_point_sample(self.mnist_points[index], 1024)

        return (
                torch.from_numpy(self.mnist_points[index]).float(),
                torch.tensor(self.mnist_targets[index]))

    def __len__(self):
        return len(self.mnist_targets)

    def __getitem__(self, index):
        return self._get_item_(index)

def show_point_cloud(tuple,seg_label=[],title=None):
    import matplotlib.pyplot as plt
    if seg_label == []:
        x = [x[0] for x in tuple]
        y = [y[1] for y in tuple]
        z = [z[2] for z in tuple]
        ax = plt.subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', cmap='spectral')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    else:
        category = list(np.unique(seg_label))
        color = ['b','r','g','y','w','b','p']
        ax = plt.subplot(111, projection='3d')
        for categ_index in range(len(category)):
            tuple_seg = tuple[seg_label == category[categ_index]]
            x = [x[0] for x in tuple_seg]
            y = [y[1] for y in tuple_seg]
            z = [z[2] for z in tuple_seg]
            ax.scatter(x, y, z, c=color[categ_index], cmap='spectral')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    import torch
    data = ModelNetDataLoader('./modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        import ipdb; ipdb.set_trace()
        print(point.shape)
        print(label.shape)
        show_point_cloud(point[0])


