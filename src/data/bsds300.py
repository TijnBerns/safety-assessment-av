import os
from pathlib import Path
import numpy as np
from base import CustomDataset, split_data
import pandas as pd
import h5py


class BSDS300Dataset(CustomDataset):
    def __init__(self, split=None, frac=None):
        self.root = Path(os.environ['DATAROOT']) / self.root
        self.path = self.root / 'bsds300.hdf5'
        
        splits = dict(zip(
            ('train', 'val', 'test'),
            self.load_data()
        ))
        
        self.data = np.array(splits[split]).astype(np.float32)
        self.n, self.dim = self.data.shape
        if frac is not None:
            self.n = int(frac * self.n)
            
    def load_data(self):
        file = h5py.File(self.path, 'r')
        return file['train'], file['validation'], file['test']

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n