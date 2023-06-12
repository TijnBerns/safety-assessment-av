from torch.utils.data import Dataset
import numpy as np
from utils import load_json
from pathlib import Path

class CustomDataset(Dataset):
    def __init__(self, root:Path, split=None) -> None:
        self.root = root
        self.name = root.name
        self.split = split
        if split is None:
            self.path = None
            self.data = None
        else:
            self.path = self.root / (split + '.npy')
            self.data = np.load(self.path)
        
        self.load_stats()

    def load_data():
        raise NotImplementedError
    
    def load_stats(self):
        try:
            stats = load_json(Path(self.root) / 'stats.json')
            self.xi = stats['Xi']
            self.threshold = stats['threshold']
            self.weight = stats['weight']
        except (FileNotFoundError, KeyError):
            print('Cannot initialize stats yet, try preprocessing data first.')
            return None
        return stats
        
    def __getitem__(self, index):
        raise self.data[index]
    
    def __len__(self):
        return len(self.data)
    

def split_data(data: np.ndarray, frac: float = 0.9):
    split_idx = int(frac * len(data))
    return data[:split_idx], data[split_idx:]


