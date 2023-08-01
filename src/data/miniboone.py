import sys
sys.path.append('src')


from pathlib import Path
import numpy as np
import os

from data.base import CustomDataset, split_data
    
class MiniBoone(CustomDataset):
    def __init__(self, split='data') -> None:
        super().__init__(Path(os.environ['DATAROOT']) / 'miniboone', split)
        
    def load_data(self):
        data = np.load(self.root / 'data.npy')
        return split_data(data)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    

