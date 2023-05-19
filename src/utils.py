import os
from typing import Tuple, Any
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json


"""Functions that check types."""
def is_bool(x):
    return isinstance(x, bool)


def is_int(x):
    return isinstance(x, int)


def is_positive_int(x):
    return is_int(x) and x > 0


def is_nonnegative_int(x):
    return is_int(x) and x >= 0


def is_power_of_two(n):
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False


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

def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not is_nonnegative_int(num_batch_dims):
        raise TypeError('Number of batch dimensions must be a non-negative integer.')
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)

def tile(x, n):
    if not is_positive_int(n):
        raise TypeError('Argument \'n\' must be a positive integer.')
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1