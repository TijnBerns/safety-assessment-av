import sys
sys.path.append('src')

import os
from pathlib import Path
import numpy as np
from data.base import CustomDataset, split_data
import pandas as pd


class Power(CustomDataset):
    def __init__(self, split=None) -> None:
        super().__init__(Path(os.environ['DATAROOT']) / 'power', split)

    def load_data(self):
        # Following https://github.com/gpapamak/maf/blob/master/datasets/power.py
        rng = np.random.RandomState(42)
        
        data = np.load(self.root / 'data.npy')
        rng.shuffle(data)
        N = data.shape[0]
        
        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)

        # Generate noise
        voltage_noise = 0.01 * rng.rand(N, 1)
        gap_noise = 0.001 * rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        
        # Add noise
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data = data + noise
        return split_data(data)
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)