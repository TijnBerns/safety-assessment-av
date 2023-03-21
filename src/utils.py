import os
from typing import Tuple
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def set_device() -> Tuple[str, str]:
    """Checks whether CUDA and SLURM are both avaible

    Returns:
        Tuple[str, str]: Device and SLURM job ID
    """
    default_jobid = "0000000"
    jobid = os.environ.get("SLURM_JOB_ID", default_jobid)
    device = 'cuda' if torch.cuda.is_available() and jobid != default_jobid else 'cpu'

    if jobid == default_jobid and device != "cpu":
        exit("Running on GPU without use of slurm!")
    elif jobid != default_jobid and device == "cpu":
        exit("Running slurm job without using GPU!")

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

    df.to_csv(path, index=False)


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
        _type_: p_edge, n_normal, and n_edge
    """
    f_split = f.split('.')
    p_edge = f_split[0][7:] + '.' + f_split[1]
    n_normal = f_split[2][9:]
    n_edge = f_split[3][7:]
    return p_edge, n_normal, n_edge

    
def plot_pdf(x_values, distribution):
    _, ax = plt.subplots(1,1)
    ax.plot(x_values, distribution.pdf(x_values))
    plt.show()
    

    

    

