
import sys
sys.path.append('src')

import os
from pathlib import Path

from data.base import CustomDataset

import h5py


class BSDS300Dataset(CustomDataset):
    def __init__(self, split=None, frac=None):
        super().__init__(Path(os.environ['DATAROOT']) / 'BSDS300', split)       
            
    def load_data(self):
        file = h5py.File(self.root / 'BSDS300.hdf5', 'r')
        return file['train'], file['validation'], file['test']

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n