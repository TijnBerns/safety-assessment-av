from torch.utils.data import Dataset
import numpy as np
from utils import save_csv, save_json, save_np
from config import FlowParameters
import os
from pathlib import Path

class CustomDataset(Dataset):
    def __init__(self) -> None:
        self.root = os.environ['DATAROOT']
        self.path = None
        self.data = None
        raise NotImplementedError
        
    def load_data():
        raise NotImplementedError
        
    def __getitem__(self, index):
        raise NotImplementedError
    

def split_data(data: np.ndarray, frac: float = 0.9):
    split_idx = int(frac * len(data))
    return data[:split_idx], data[split_idx:]


