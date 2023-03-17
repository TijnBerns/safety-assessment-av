import os
from typing import Tuple
import torch
from pathlib import Path
import pandas as pd


def set_device() -> Tuple[str, str]:
    default_jobid = "0000000"
    jobid = os.environ.get("SLURM_JOB_ID", default_jobid)
    device = 'cuda' if torch.cuda.is_available() and jobid != default_jobid else 'cpu'

    if jobid == default_jobid and device != "cpu":
        exit("Running on GPU without use of slurm!")
    elif jobid != default_jobid and device == "cpu":
        exit("Running slurm job without using GPU!")

    return device, jobid


def save_csv(path: Path, df: pd.DataFrame):
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True)

    df.to_csv(path)
