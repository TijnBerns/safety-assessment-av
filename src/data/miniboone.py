import sys
sys.path.append('src')

from utils import save_json, save_np
from typing import Union
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import os
from config import FlowParameters
from scipy.optimize import fsolve
from data.base import CustomDataset, split_data
    
class MiniBoone(CustomDataset):
    def __init__(self, split='data') -> None:
        self.root = Path(os.environ['DATAROOT']) / 'miniboone'
        self.split = split
        if split is None:
            self.path = None
            self.data = None
        else:
            self.path = self.root / (split + '.npy')
            self.data = np.load(self.path)
        
    def load_data(self):
        data = np.load(self.root / 'data.npy')
        return split_data(data)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    

