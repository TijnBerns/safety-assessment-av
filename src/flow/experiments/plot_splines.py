import sys

sys.path.append("src")
sys.path.append("src/flow")

import numpy as np
import scipy.stats
from pprint import pprint
from datetime import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader

import pandas as pd
from nflows import flows


import click

from flow.flow_module import FlowModule
import flow.evaluate


# from flow_module import create_flow
import flow.parameters as parameters
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "8"


import utils
from pathlib import Path


def main(version):
    ckpt, _ = flow.evaluate.get_pl_checkpoint(version)
    ckpt = flow.evaluate.get_best_checkpoint(ckpt)

    evaluator = flow.evaluate.Evaluator("gas")
    flow_module = evaluator._load_checkpoint(ckpt)
    flow: flows.Flow = flow_module.flow

    breakpoint()


if __name__ == "__main__":
    main(253784)
