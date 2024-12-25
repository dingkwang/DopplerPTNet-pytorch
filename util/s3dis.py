import os
import numpy as np
from torch.utils.data import Dataset
from util.data_util import data_prepare

class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, 
                 voxel_max=None, transform=None, shuffle_index=False, loop=1,
                 velocity=True):
        super().__init__()
        self.split = split
        self.voxel_size = voxel_size
        self.transform = transform
        self.voxel_max = voxel_max
        self.shuffle_index = shuffle_index
        self.loop = loop
        self.velocity = velocity
        
        # Get list of data files
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        
        # Split data based on test area
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
            
        # Store the data paths
        self.data_paths = [os.path.join(data_root, item + '.npy') for item in self.data_list]
        
        # Create index array
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        # Get data index accounting for loop
        data_idx = self.data_idx[idx % len(self.data_idx)]
        # Load data directly from file
        data = np.load(self.data_paths[data_idx])  # xyzrgbl, N*7
        
        # Split into coordinates, features, and labels
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        
        # Process the data
        coord, feat, label = data_prepare(coord, feat, label, self.split, 
                                        self.voxel_size, self.voxel_max, 
                                        self.transform, self.shuffle_index)
        if self.velocity:
            feat = feat[:, :1]
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop