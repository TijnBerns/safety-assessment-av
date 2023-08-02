"""
Module containing BSDS300 dataset class used in normalizing flow experiments.
"""

import os
from pathlib import Path
from base import CustomDataset
import h5py


class BSDS300Dataset(CustomDataset):
    def __init__(self, split=None, frac=None):
        super().__init__(Path(os.environ["DATAROOT"]) / "BSDS300", split)
        self.dataset_str = "bsds300"

    def load_data(self):
        file = h5py.File(self.root / "BSDS300.hdf5", "r")
        return file["train"], file["validation"], file["test"]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
