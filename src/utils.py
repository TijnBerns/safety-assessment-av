import os
from typing import Tuple, Any, Union
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import numpy as np
import torch 
import random

# """Functions that check types."""
# def is_bool(x):
#     return isinstance(x, bool)


# def is_int(x):
#     return isinstance(x, int)


# def is_positive_int(x):
#     return is_int(x) and x > 0


# def is_nonnegative_int(x):
#     return is_int(x) and x >= 0


# def is_power_of_two(n):
#     if is_positive_int(n):
#         return not n & (n - 1)
#     else:
#         return False

# Constants
FIGSIZE_1_1 = (6,2)
FIGSIZE_1_2 = (3., 1.8)
FIGSIZE_1_3 = (2.0, 2.0)


def seed_all(seed: int):
    torch.manual_seed(seed) 
    random.seed(seed)
    np.random.seed(seed)
    

def set_device() -> Tuple[str, str]:
    """Checks whether CUDA and SLURM are both avaible

    Returns:
        Tuple[str, str]: Device and SLURM job ID
    """
    default_jobid = "0000000"
    jobid = os.environ.get("SLURM_JOB_ID", default_jobid)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # if jobid == default_jobid and device != "cpu":
    #     exit("Running on GPU without use of slurm!")

    return device, jobid


def save_np(path: Union[str, Path], data: np.ndarray):
    path = str(path)
    with open(path, 'wb') as f:
        np.save(f, data)

def save_csv(path: Path, df: pd.DataFrame):
    """Save a dataframe as csv to defined path

    Args:
        path (Path): Path where dataframe will be saved
        df (pd.DataFrame): The dataframe 
    """
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True)

    df.to_csv(path, index=False, lineterminator='\n', sep=',')
    
def save_json(path: Path, data: Any):
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True)

    with open(path, 'w') as f:
        json.dump(data, f)
        
def load_json(path: Path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data  
        
def load_json_as_df(path: Path):
    with open(path,'r') as f:
        data = json.load(f)
        
    return pd.DataFrame(data) 

def rec_dd():
    """Recursive defaultdict

    Returns:
        _type_: defaultdict
    """
    return defaultdict(rec_dd)    


def variables_from_filename(f: str):
    """Extracts the variables from filename     

    Args:
        f (str): The file name structures as p_edge_X.n_normal_Y.n_edge_Z

    Returns:
        Tuple[float]: p_edge, n_normal, n_edge, corr
    """
    f_split = f.split('.')
    p_edge = float(f_split[0][7:] + '.' + f_split[1])
    n_normal = int(f_split[2][9:])
    n_edge = int(f_split[3][7:])
    corr = float(f_split[4][5:] + '.' + f_split[5])
    return p_edge, n_normal, n_edge, corr

def create_alternating_binary_mask(features, even=True):
    """
    Creates a binary mask of a given dimension which alternates its masking.

    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask
