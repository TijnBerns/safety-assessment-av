"""
Module containing GAS dataset class used in normalizing flow experiments.
"""

import sys

sys.path.append("src")

import os
from pathlib import Path
import numpy as np
from data.base import CustomDataset, split_data
import pandas as pd
import pickle


class Gas(CustomDataset):
    def __init__(self, split=None) -> None:
        super().__init__(Path(os.environ["DATAROOT"]) / "gas", split)

    def load_data(self):
        # Following https://github.com/gpapamak/maf/blob/master/datasets/gas.py

        data = pd.read_pickle(self.root / "ethylene_CO.pickle")
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)

        # Get correlation numbers
        def get_correlation_numbers(data):
            C = data.corr()
            A = C > 0.98
            B = A.values.sum(axis=1)
            return B

        # Clean data
        B = get_correlation_numbers(data)
        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = get_correlation_numbers(data)

        data = data.values
        return split_data(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
