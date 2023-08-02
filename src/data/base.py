"""
Module containing base dataset class used in normalizing flow experiments.
"""

import numpy as np
from pathlib import Path
from utils import load_json
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root: Path, split=None) -> None:
        self.root = root
        self.name = root.name
        self.split = split
        if split is None:
            self.path = None
            self.data = None
        else:
            self.path = self.root / (split + ".npy")
            self.data = np.load(self.path)

        self.load_stats()

    def load_data(self):
        raise NotImplementedError

    def load_stats(self):
        try:
            self.stats = load_json(Path(self.root) / "stats.json")
            self.xi = self.stats["Xi"]
            self.threshold = self.stats["threshold"]
            self._threshold = self.stats["_threshold"]
            self.weight = self.stats["weight"]
        except (FileNotFoundError, KeyError):
            print("Cannot initialize stats yet, try preprocessing data first.")
            return None
        return self.stats

    def __getitem__(self, index):
        raise self.data[index]

    def __len__(self):
        return len(self.data)


def split_data(data: np.ndarray, frac: float = 0.9):
    """Splits data provided in np array into two parts based on provided fraction.

    Args:
        data (np.ndarray): Data that is split.
        frac (float, optional): Fraction of data that is kept in the first dataset split. 
        Defaults to 0.9.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two splits, one containing frac percent of data, one 
        containing (1-frac) percent of data
    """
    split_idx = int(frac * len(data))
    return data[:split_idx], data[split_idx:]
